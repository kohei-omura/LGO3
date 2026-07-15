#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GACHA ORACLE 暦データ自動更新スクリプト v2
==========================================
ajnet.ne.jp の暦カレンダーから「六曜・月齢・旧暦」を取得し calendar.json を生成。
GitHub Actions で定期実行 → 各アプリ(GACHA ORACLE / LGO3 等)が起動時に読み込み、
内蔵の較正計算より優先して使う。取得失敗・範囲外日は各アプリ側の計算へフォールバック。

v2 の変更点
-----------
1. 常に「今日 +365日」以上を確保（v1 は当月から6ヶ月固定＝最短151日まで痩せた）
2. 全置換 → マージ方式。失敗した月は既存値を保持するため、穴が開かない
3. 閏月バグ修正: 取得元は閏月を「旧暦 ※閏5/9」と表記するため、
   v1 の正規表現 (\\d+)/(\\d+) が数字直結を要求して閏月の全日を黙って捨てていた
   （2028年閏5月＝29日分が消失していた）
4. 3段の検証ゲートを追加（1日単位で判定し、通らない日は採用しない）
   ①六曜が (旧暦月+旧暦日-2)%6 の法則と一致するか（セル内容の整合性）
   ②取得元が併記する干支が60干支循環の計算値と一致するか（日付キー自体の整合性）
   ③月齢が前日比 +1前後 か、朔で0に戻っているか（連続性）
5. 差分取得: 直近2ヶ月は毎回再取得（朔のズレ補正）、それ以外は未取得の月のみ。
   FORCE_FULL=1 で全期間を再取得
6. 過去60日分を保持（履歴表示から過去日の六曜を引けるようにするため）

環境変数
--------
FORCE_FULL=1     全期間を強制再取得
GITHUB_OUTPUT    設定時、coverage_days / ok を書き出す（Actions の検証ステップ用）
"""
import urllib.request, urllib.parse, re, json, datetime, sys, os, time, calendar

CAL_URL = ("https://www.ajnet.ne.jp/diary_f/"
           "?ohGmI01LrhInI9dh0n9433nHyCnnmTYJSZrkYzAGNqVw3XS8nNRJcIybHViQAayndyWtJjF7IP7aGnHOX3ExN0cE1Yt9BrRoGavGNE1RjYQuPqnkjmFC1kQFVP9NInI9")
UA   = {'User-Agent': 'Mozilla/5.0 (GachaOracle CalendarUpdater/2.0)'}
OUT  = 'calendar.json'

DAYS_AHEAD     = 400   # 今日から先に確保する日数（365日を安全マージン付きで満たす）
DAYS_REQUIRED  = 365   # これを下回ったら Actions を失敗させて通知する
RETAIN_PAST    = 60    # 過去何日分を残すか（履歴表示用）
REFRESH_MONTHS = 2     # 先頭から何ヶ月を毎回再取得するか（朔のズレ補正）
POLITE_SLEEP   = 1.5   # 取得元への負荷を避けるための待機秒数

ROKUYO = ['先勝', '友引', '先負', '仏滅', '大安', '赤口']
STEMS    = '甲乙丙丁戊己庚辛壬癸'
BRANCHES = '子丑寅卯辰巳午未申酉戌亥'

JST = datetime.timezone(datetime.timedelta(hours=9))


# ────────────────────────────────────────────
#  検証ヘルパ
# ────────────────────────────────────────────
def kanshi_of(d: datetime.date) -> str:
    """その日の干支。検証アンカー: 1970-01-01 = 辛巳"""
    ep = (d - datetime.date(1970, 1, 1)).days
    n  = ((ep + 17) % 60 + 60) % 60
    return STEMS[n % 10] + BRANCHES[n % 12]


def rokuyo_of_lunar(lunar_month: int, lunar_day: int) -> str:
    """六曜の定義: (旧暦月 + 旧暦日 - 2) mod 6。閏月は元の月番号を使う"""
    return ROKUYO[(lunar_month + lunar_day - 2) % 6]


# ────────────────────────────────────────────
#  取得
# ────────────────────────────────────────────
# 六曜(干支) ... 旧暦 [※閏]M/D ... 月齢 X.X
RE_CELL_ROKUYO = re.compile(r'(先勝|友引|先負|仏滅|大安|赤口)(?:&nbsp;)*\((.)(.)\)')
RE_CELL_LUNAR  = re.compile(r'旧暦&nbsp;(※?閏)?(\d+)/(\d+)')   # ← 閏月対応（v1 のバグ修正箇所）
RE_CELL_MOON   = re.compile(r'月齢&nbsp;([\d.]+)')


def fetch_month(year, month):
    """指定年月をPOST取得しパース → ({'YYYY-MM-DD': {...}}, rejected_count)"""
    data = urllib.parse.urlencode({'yy': str(year), 'mm': str(month), 'send': '変更'}).encode('euc-jp')
    req  = urllib.request.Request(CAL_URL, data=data, headers=UA)
    html = urllib.request.urlopen(req, timeout=30).read().decode('euc-jp', errors='replace')

    result, rejected = {}, 0
    # セルは <b>(&nbsp;)?DD</b> を境に並ぶ
    cells = re.split(r'<b>(?:&nbsp;)?(\d{1,2})</b>', html)
    for i in range(1, len(cells) - 1, 2):
        try:
            day = int(cells[i])
            d   = datetime.date(year, month, day)
        except ValueError:
            continue
        body = re.sub(r'<[^>]+>', '', cells[i + 1])   # 六曜と干支の間にタグが挟まるため先に除去
        rk = RE_CELL_ROKUYO.search(body)
        ml = RE_CELL_LUNAR.search(body)
        mm = RE_CELL_MOON.search(body)
        if not (rk and ml and mm):
            continue

        rokuyo   = rk.group(1)
        site_kan = rk.group(2) + rk.group(3)
        is_leap  = bool(ml.group(1))
        lm, ld   = int(ml.group(2)), int(ml.group(3))
        moon_age = float(mm.group(1))

        # ── 検証ゲート① セル内容の整合性: 六曜 ↔ 旧暦
        if rokuyo_of_lunar(lm, ld) != rokuyo:
            print(f"  [reject] {d} 六曜↔旧暦不一致: {rokuyo} vs 旧暦{lm}/{ld}", file=sys.stderr)
            rejected += 1
            continue
        # ── 検証ゲート② 日付キーの整合性: 併記の干支 ↔ 60干支循環
        if site_kan != kanshi_of(d):
            print(f"  [reject] {d} 干支不一致: サイト{site_kan} vs 計算{kanshi_of(d)}", file=sys.stderr)
            rejected += 1
            continue
        # ── 値域チェック
        if not (0.0 <= moon_age <= 30.5) or not (1 <= lm <= 12) or not (1 <= ld <= 30):
            print(f"  [reject] {d} 値域外: 月齢{moon_age} 旧暦{lm}/{ld}", file=sys.stderr)
            rejected += 1
            continue

        result[d.isoformat()] = {
            'rokuyo' : rokuyo,
            'moonAge': moon_age,
            'lunar'  : ('閏' if is_leap else '') + f"{lm}/{ld}",
        }
    return result, rejected


# ────────────────────────────────────────────
#  マージ後の連続性チェック（検証ゲート③）
# ────────────────────────────────────────────
def check_continuity(days):
    """月齢が前日比 +1前後 / 朔で0に戻る 以外を異常として列挙"""
    bad = []
    ks = sorted(days)
    for a, b in zip(ks, ks[1:]):
        da, db = datetime.date.fromisoformat(a), datetime.date.fromisoformat(b)
        if (db - da).days != 1:
            continue                      # 日付が飛んでいる箇所は連続性の対象外
        diff = days[b]['moonAge'] - days[a]['moonAge']
        if not (0.5 < diff < 1.5) and not (diff < -25):
            bad.append((a, b, round(diff, 2)))
    return bad


def gh_output(**kv):
    p = os.environ.get('GITHUB_OUTPUT')
    if not p:
        return
    with open(p, 'a', encoding='utf-8') as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


# ────────────────────────────────────────────
#  メイン
# ────────────────────────────────────────────
def main():
    today      = datetime.datetime.now(JST).date()
    force_full = os.environ.get('FORCE_FULL') == '1'
    horizon    = today + datetime.timedelta(days=DAYS_AHEAD)
    keep_from  = today - datetime.timedelta(days=RETAIN_PAST)

    # ① 既存データを読み込む（マージのベース。失敗した月はここの値が残る）
    existing = {}
    try:
        with open(OUT, encoding='utf-8') as f:
            existing = json.load(f).get('days', {})
        print(f"[base] 既存 {len(existing)}日分を読み込み", file=sys.stderr)
    except Exception as ex:
        print(f"[base] 既存なし（新規生成）: {ex}", file=sys.stderr)

    # ② 今日〜horizon を覆う月を列挙
    months = []
    y, m = today.year, today.month
    while (y, m) <= (horizon.year, horizon.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    # ③ 差分取得: 直近REFRESH_MONTHSヶ月は毎回 / それ以外は未充足の月のみ
    merged = dict(existing)
    ok_months = fetched = total_rejected = 0
    for idx, (yy, mm) in enumerate(months):
        dim = calendar.monthrange(yy, mm)[1]
        have = sum(1 for d in range(1, dim + 1)
                   if f"{yy:04d}-{mm:02d}-{d:02d}" in existing)
        need = force_full or idx < REFRESH_MONTHS or have < dim
        if not need:
            ok_months += 1
            continue

        if fetched:
            time.sleep(POLITE_SLEEP)
        fetched += 1
        try:
            md, rej = fetch_month(yy, mm)
            total_rejected += rej
            if md:
                merged.update(md)
                ok_months += 1
                mark = '' if len(md) == dim else f" ⚠{dim - len(md)}日欠（取得元の収録上限付近の可能性）"
                print(f"[ok]   {yy}-{mm:02d}: {len(md)}/{dim}日{mark}", file=sys.stderr)
            else:
                print(f"[warn] {yy}-{mm:02d}: 有効日ゼロ → 既存値を保持", file=sys.stderr)
        except Exception as ex:
            print(f"[warn] {yy}-{mm:02d} 取得失敗 → 既存値を保持: {ex}", file=sys.stderr)

    if not merged:
        print("[error] データ皆無。既存 calendar.json を維持して終了。", file=sys.stderr)
        gh_output(coverage_days=0, ok='false')
        sys.exit(0)

    # ④ 検証ゲート③: 連続性の異常な日は採用しない
    for a, b, diff in check_continuity(merged):
        print(f"  [reject] {b} 月齢が不連続 ({a}比 {diff:+}) → 除外", file=sys.stderr)
        merged.pop(b, None)
        total_rejected += 1

    # ⑤ 保持期間外を刈る
    pruned = {k: v for k, v in merged.items()
              if keep_from.isoformat() <= k <= horizon.isoformat()}

    # ⑥ カバレッジ判定（今日から連続して何日先まであるか）
    coverage = 0
    d = today
    while d.isoformat() in pruned:
        coverage += 1
        d += datetime.timedelta(days=1)
    coverage_ahead = coverage - 1 if coverage else -1

    out = {
        'updated' : datetime.datetime.now(JST).strftime('%Y-%m-%d %H:%M JST'),
        'source'  : 'ajnet.ne.jp',
        'coverage': {
            'from'      : min(pruned) if pruned else None,
            'to'        : max(pruned) if pruned else None,
            'daysAhead' : coverage_ahead,
            'required'  : DAYS_REQUIRED,
        },
        'days': dict(sorted(pruned.items())),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    good = coverage_ahead >= DAYS_REQUIRED
    print(f"[done] {len(pruned)}日分を {OUT} に保存 "
          f"({out['coverage']['from']} → {out['coverage']['to']})")
    print(f"[done] 今日から連続 {coverage_ahead}日先まで確保 / 必要 {DAYS_REQUIRED}日 "
          f"{'✅' if good else '❌'} / 取得 {fetched}ヶ月 / 不採用 {total_rejected}日")
    gh_output(coverage_days=coverage_ahead, ok='true' if good else 'false')


if __name__ == '__main__':
    main()
