"""
app.py - 2026 World Cup Predictions Dashboard.
ML model vs. fan bracket, both locked before kickoff.
"""
import json, math
from pathlib import Path
import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go

ROOT = Path(__file__).parent
with open(sorted((ROOT / "predictions").glob("dual_bracket_*.json"))[-1]) as f:
    locked = json.load(f)
with open(ROOT / "data" / "processed" / "predictions_adjusted.json") as f:
    detailed = json.load(f)
with open(ROOT / "data" / "processed" / "backtest_results.json") as f:
    backtest = json.load(f)
with open(ROOT / "data" / "processed" / "calibration.json") as f:
    calibration = json.load(f)
with open(ROOT / "data" / "processed" / "bookmaker_comparison.json") as f:
    bookmaker = json.load(f)

ml_bracket = locked["ml_model"]

# Reconstruct ML SF teams to include both finalists.
# The locked file computed SF (top 4 by P(SF)) and finalists (top 2 by P(final))
# independently, which produced an impossible bracket where a finalist (France)
# was not in the top-4 SF list. Display-only fix; locked predictions unchanged.
_ml_finalists = list(ml_bracket["finalists"])
_p_sf_sorted = sorted(locked["ml_full_probs"], key=lambda x: -x["p_sf"])
_other_sf = [t["team"] for t in _p_sf_sorted
             if t["team"] not in _ml_finalists][:max(0, 4 - len(_ml_finalists))]
ml_bracket = dict(ml_bracket)
ml_bracket["sf_teams"] = _ml_finalists + _other_sf
fan_bracket = locked["fan_bracket"]
ml_probs = sorted(locked["ml_full_probs"], key=lambda x: x["p_win"], reverse=True)
match_preds = detailed["match_predictions"]
team_probs = {t["team"]: t for t in ml_probs}

for m in match_preds:
    ps = [m["p_home_win"], m["p_draw"], m["p_away_win"]]
    m["entropy"] = -sum(p * math.log(p) for p in ps if p > 0)

FLAGS = {
    "Mexico":"🇲🇽","South Africa":"🇿🇦","South Korea":"🇰🇷","Czech Republic":"🇨🇿",
    "Canada":"🇨🇦","Bosnia and Herzegovina":"🇧🇦","United States":"🇺🇸","Paraguay":"🇵🇾",
    "Haiti":"🇭🇹","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Australia":"🇦🇺","Turkey":"🇹🇷",
    "Brazil":"🇧🇷","Morocco":"🇲🇦","Qatar":"🇶🇦","Switzerland":"🇨🇭",
    "Ivory Coast":"🇨🇮","Ecuador":"🇪🇨","Germany":"🇩🇪","Curaçao":"🇨🇼",
    "Netherlands":"🇳🇱","Japan":"🇯🇵","Sweden":"🇸🇪","Tunisia":"🇹🇳",
    "Saudi Arabia":"🇸🇦","Uruguay":"🇺🇾","Spain":"🇪🇸","Cape Verde":"🇨🇻",
    "Iran":"🇮🇷","New Zealand":"🇳🇿","Belgium":"🇧🇪","Egypt":"🇪🇬",
    "France":"🇫🇷","Senegal":"🇸🇳","Iraq":"🇮🇶","Norway":"🇳🇴",
    "Argentina":"🇦🇷","Algeria":"🇩🇿","Austria":"🇦🇹","Jordan":"🇯🇴",
    "Ghana":"🇬🇭","Panama":"🇵🇦","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Croatia":"🇭🇷",
    "Portugal":"🇵🇹","DR Congo":"🇨🇩","Uzbekistan":"🇺🇿","Colombia":"🇨🇴",
}
def fl(t): return FLAGS.get(t, "🏳")

ML_C, FAN_C = "#1e3a8a", "#a16207"
MUTED, BORDER, BG_LIGHT = "#64748b", "#e2e8f0", "#f8fafc"
TEXT, CARD = "#0f172a", "#ffffff"

# ──── ML top-10 ────
top10 = ml_probs[:10]
fig_top10 = go.Figure(go.Bar(
    x=[t["p_win"]*100 for t in top10][::-1],
    y=[f"{fl(t['team'])}  {t['team']}" for t in top10][::-1],
    orientation='h',
    marker=dict(color=[ML_C if i == len(top10)-1 else "#3b82f6" for i in range(len(top10))]),
    text=[f"<b>{t['p_win']*100:.1f}%</b>" for t in top10][::-1],
    textposition='auto', textfont=dict(size=13, color=TEXT),
    hovertemplate="<b>%{y}</b><br>%{x:.1f}% chance to win<extra></extra>"
))
fig_top10.update_layout(
    xaxis=dict(title="Win probability (%)", showgrid=True, gridcolor="#f1f5f9", zeroline=False),
    yaxis=dict(tickfont=dict(size=13), automargin=True),
    height=460, margin=dict(l=10, r=30, t=30, b=60), autosize=True,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT),
)

# ──── Fan bracket depth chart ────
fan_qf = set(fan_bracket["qf_teams"])
fan_sf = set(fan_bracket["sf_teams"])
fan_finalists = set(fan_bracket["finalists"])
fan_champion = fan_bracket["champion"]
depth = {}
for t in fan_qf: depth[t] = 3
for t in fan_sf: depth[t] = 4
for t in fan_finalists: depth[t] = 5
depth[fan_champion] = 6
fan_picks_sorted = sorted(depth.items(), key=lambda x: -x[1])
STAGE_LBL = {3: "Quarter-final", 4: "Semi-final", 5: "Finalist", 6: "Champion"}
STAGE_COLORS = {6: "#92400e", 5: "#b45309", 4: "#d97706", 3: "#f59e0b"}

fig_fan = go.Figure(go.Bar(
    x=[d[1] for d in fan_picks_sorted][::-1],
    y=[f"{fl(d[0])}  {d[0]}" for d in fan_picks_sorted][::-1],
    orientation='h',
    marker=dict(color=[STAGE_COLORS.get(d[1], "#fbbf24") for d in fan_picks_sorted][::-1]),
    text=[STAGE_LBL[d[1]] for d in fan_picks_sorted][::-1],
    textposition='none', textfont=dict(size=13, color=TEXT),
    hovertemplate="<b>%{y}</b><br>Picked to reach: %{text}<extra></extra>"
))
fig_fan.update_layout(
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 8]),
    yaxis=dict(tickfont=dict(size=13), automargin=True),
    height=460, margin=dict(l=20, r=160, t=30, b=30),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT),
)

# ──── Per-match table ────
match_table = []
for m in sorted(match_preds, key=lambda x: (x["date"], x["match_id"])):
    ph, pdw, pa = m["p_home_win"]*100, m["p_draw"]*100, m["p_away_win"]*100
    match_table.append({
        "Date": m["date"][5:], "G": m["group"],
        "Match": f"{fl(m['home_team'])} {m['home_team']}  vs  {m['away_team']} {fl(m['away_team'])}",
        "Home": f"{ph:.0f}%", "Draw": f"{pdw:.0f}%", "Away": f"{pa:.0f}%",
        "xG_H": f"{m['expected_home_goals']:.2f}",
        "xG_A": f"{m['expected_away_goals']:.2f}",
        "Likely": m["most_likely_score"],
    })

# ──── Groups ────
group_teams = {}
for m in match_preds:
    g = m["group"]
    if g not in group_teams: group_teams[g] = set()
    group_teams[g].add(m["home_team"]); group_teams[g].add(m["away_team"])

def make_group_card(g):
    teams = sorted(group_teams[g], key=lambda t: -team_probs.get(t, {}).get("elo_adj", 1500))
    rows = []
    for i, t in enumerate(teams):
        p_adv = team_probs.get(t, {}).get("p_r32", 0) * 100
        rows.append(html.Tr([
            html.Td(str(i+1), style={"color": MUTED, "padding": "6px 8px", "width": "20px", "fontSize": "0.85em"}),
            html.Td([html.Span(fl(t), style={"marginRight": "6px"}), t], style={"padding": "6px 8px"}),
            html.Td(f"{p_adv:.0f}%", style={"padding": "6px 8px", "textAlign": "right",
                    "color": ML_C if p_adv > 50 else MUTED,
                    "fontWeight": "600" if p_adv > 50 else "normal"}),
        ]))
    return html.Div(
        style={"background": CARD, "padding": "18px 20px", "borderRadius": "10px",
               "border": f"1px solid {BORDER}", "flex": "1 1 240px", "minWidth": "240px"},
        children=[
            html.Div(f"GROUP {g}", style={"fontSize": "0.7em", "color": MUTED,
                "letterSpacing": "0.08em", "fontWeight": "700", "marginBottom": "10px"}),
            html.Table(style={"width": "100%", "fontSize": "0.92em", "borderCollapse": "collapse"},
                children=[html.Tbody(rows)])
        ]
    )

# ──── Most uncertain matches ────
most_uncertain = sorted(match_preds, key=lambda x: -x["entropy"])[:6]
uncertain_cards = []
for m in most_uncertain:
    ph, pdw, pa = m["p_home_win"]*100, m["p_draw"]*100, m["p_away_win"]*100
    uncertain_cards.append(html.Div(
        style={"background": CARD, "padding": "18px", "borderRadius": "10px",
               "border": f"1px solid {BORDER}", "flex": "1 1 300px", "minWidth": "300px"},
        children=[
            html.Div(m["date"], style={"fontSize": "0.75em", "color": MUTED, "marginBottom": "6px",
                    "letterSpacing": "0.05em"}),
            html.Div([html.Span([fl(m['home_team']), " ", html.Strong(m['home_team'])]),
                     html.Span(" vs ", style={"color": MUTED, "margin": "0 6px"}),
                     html.Span([html.Strong(m['away_team']), " ", fl(m['away_team'])]),
                     ], style={"fontSize": "1.02em", "marginBottom": "12px"}),
            html.Div(style={"display": "flex", "gap": "4px", "marginBottom": "8px", "height": "24px"},
                children=[
                    html.Div(f"{ph:.0f}%", style={"flex": ph, "background": ML_C, "color": "white",
                        "fontSize": "0.78em", "borderRadius": "3px", "textAlign": "center",
                        "lineHeight": "24px"}),
                    html.Div(f"D {pdw:.0f}%", style={"flex": pdw, "background": MUTED, "color": "white",
                        "fontSize": "0.78em", "borderRadius": "3px", "textAlign": "center",
                        "lineHeight": "24px"}),
                    html.Div(f"{pa:.0f}%", style={"flex": pa, "background": "#7c3aed", "color": "white",
                        "fontSize": "0.78em", "borderRadius": "3px", "textAlign": "center",
                        "lineHeight": "24px"}),
                ]),
            html.Div(f"Most likely score: {m['most_likely_score']}",
                style={"fontSize": "0.85em", "color": MUTED}),
        ]))

# ──── APP ────

def make_backtest_card(yd):
    winner, runner_up = yd["winner"], yd["runner_up"]
    rows = []
    for i, r in enumerate(yd["top10"][:5]):
        marker = ""
        if r["team"] == winner: marker = "actual winner"
        elif r["team"] == runner_up: marker = "actual runner-up"
        rows.append(html.Tr([
            html.Td(str(i+1), style={"color": MUTED, "padding": "6px 8px", "width": "24px", "fontSize": "0.85em"}),
            html.Td([fl(r["team"]), " ", r["team"]], style={"padding": "6px 8px"}),
            html.Td(f"{r['p_win']*100:.1f}%", style={"padding": "6px 8px", "textAlign": "right", "color": ML_C, "fontWeight": "600"}),
            html.Td(marker, style={"padding": "6px 8px", "color": FAN_C, "fontSize": "0.8em", "fontStyle": "italic"}),
        ]))
    return html.Div(
        style={"background": CARD, "padding": "24px", "borderRadius": "12px",
               "border": f"1px solid {BORDER}", "flex": "1 1 460px", "minWidth": "420px"},
        children=[
            html.Div(yd["name"], style={"fontSize": "0.7em", "color": MUTED,
                "letterSpacing": "0.08em", "fontWeight": "700", "marginBottom": "10px"}),
            html.H3([fl(winner), " ", winner, " won"], style={"margin": "0 0 16px 0", "fontSize": "1.25em"}),
            html.Table(style={"width": "100%", "fontSize": "0.92em", "borderCollapse": "collapse", "marginBottom": "16px"},
                children=[html.Tbody(rows)]),
            html.Div(style={"borderTop": f"1px solid {BORDER}", "paddingTop": "12px", "fontSize": "0.85em", "color": "#334155", "lineHeight": "1.6"},
                children=[
                    html.Div([html.Strong(winner), f" ranked #{yd['winner_rank']} (of 32) by the model."]),
                    html.Div([html.Strong(runner_up), f" ranked #{yd['runner_up_rank']}."], style={"marginTop": "4px"}),
                    html.Div(["Log loss ", html.Strong(f"{yd['log_loss']:.3f}"), f" vs baseline {yd['baseline_log_loss']:.3f} (",
                              html.Strong(f"+{yd['improvement_pct']:.1f}%"), " improvement)"], style={"marginTop": "4px"}),
                ]),
        ]
    )


calib_data = calibration["calibration"]
calib_pred = [c["avg_predicted"]*100 for c in calib_data]
calib_actual = [c["avg_actual"]*100 for c in calib_data]
calib_n = [c["n"] for c in calib_data]

fig_calib = go.Figure()
fig_calib.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100], mode="lines",
    line=dict(color="#cbd5e1", dash="dash", width=2),
    hoverinfo="skip", showlegend=False
))
fig_calib.add_trace(go.Scatter(
    x=calib_pred, y=calib_actual, mode="markers+lines",
    marker=dict(size=[max(12, (n**0.5)*1.8) for n in calib_n],
                color=ML_C, opacity=0.85, line=dict(color="white", width=2)),
    line=dict(color=ML_C, width=2),
    text=[f"n = {n}" for n in calib_n],
    hovertemplate="Predicted: %{x:.1f}%<br>Actual: %{y:.1f}%<br>%{text}<extra></extra>",
    showlegend=False,
))
fig_calib.update_layout(
    xaxis=dict(title="Model's predicted probability (%)", range=[-3, 103],
               showgrid=True, gridcolor="#f1f5f9", zeroline=False),
    yaxis=dict(title="Actual frequency (%)", range=[-3, 103],
               showgrid=True, gridcolor="#f1f5f9", zeroline=False),
    height=440, margin=dict(l=50, r=20, t=30, b=60), autosize=True,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT),
)



top12 = bookmaker["comparison"][:12]
teams_mvm = [f"{fl(t['team'])}  {t['team']}" for t in top12][::-1]
market_vals = [t["market_pct"] for t in top12][::-1]
model_vals = [t["model_pct"] for t in top12][::-1]

fig_market = go.Figure()
fig_market.add_trace(go.Bar(
    y=teams_mvm, x=market_vals, name="Market (FanDuel)",
    orientation='h',
    marker=dict(color=FAN_C, opacity=0.9),
    text=[f"{v:.1f}%" for v in market_vals], textposition='auto',
    hovertemplate="<b>%{y}</b><br>Market: %{x:.2f}%<extra></extra>",
))
fig_market.add_trace(go.Bar(
    y=teams_mvm, x=model_vals, name="My ML model",
    orientation='h',
    marker=dict(color=ML_C, opacity=0.9),
    text=[f"{v:.1f}%" for v in model_vals], textposition='auto',
    hovertemplate="<b>%{y}</b><br>Model: %{x:.2f}%<extra></extra>",
))
fig_market.update_layout(
    barmode='group',
    xaxis=dict(title="Probability to win the tournament (%)", showgrid=True, gridcolor="#f1f5f9"),
    yaxis=dict(tickfont=dict(size=14), automargin=True),
    height=560, margin=dict(l=10, r=30, t=30, b=60), autosize=True,
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT),
)


app = dash.Dash(__name__, title="2026 World Cup Predictions",
                external_stylesheets=["https://rsms.me/inter/inter.css"])
server = app.server

def section_title(text, subtitle=None):
    return html.Div(style={"marginTop": "56px", "marginBottom": "20px"}, children=[
        html.H2(text, style={"fontSize": "1.7em", "fontWeight": "700", "color": TEXT,
                "marginBottom": "4px", "letterSpacing": "-0.01em"}),
        html.Div(subtitle, style={"color": MUTED, "fontSize": "0.95em"}) if subtitle else None,
    ])

def champion_card(label, team, subtitle, color, bg):
    return html.Div(style={"flex": 1, "padding": "32px", "borderRadius": "14px",
            "background": bg, "border": f"1px solid {color}22"},
        children=[
            html.Div(label, style={"fontSize": "0.72em", "color": color,
                "letterSpacing": "0.1em", "fontWeight": "700"}),
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "16px", "marginTop": "12px"},
                children=[html.Span(fl(team), style={"fontSize": "2.5em", "lineHeight": "1"}),
                          html.H2(team, style={"margin": 0, "fontSize": "2.2em", "fontWeight": "700", "color": color})]),
            html.Div(subtitle, style={"color": MUTED, "marginTop": "8px"}),
        ])

def stage_row(label, ml_val, fan_val):
    return html.Tr([
        html.Td(label, style={"padding": "14px 20px", "fontWeight": "600", "borderTop": f"1px solid {BORDER}"}),
        html.Td(ml_val, style={"padding": "14px 20px", "borderTop": f"1px solid {BORDER}", "fontSize": "0.95em"}),
        html.Td(fan_val, style={"padding": "14px 20px", "borderTop": f"1px solid {BORDER}", "fontSize": "0.95em"}),
    ])

def team_list(teams): return " · ".join([f"{fl(t)} {t}" for t in teams])

def critique_card(label, label_color, title_parts, body):
    return html.Div(style={"background": "white", "padding": "24px", "borderRadius": "12px",
        "border": f"1px solid {BORDER}", "borderLeft": f"4px solid {label_color}",
        "flex": "1 1 300px", "minWidth": "300px"},
        children=[
            html.Div(label, style={"fontSize": "0.7em", "color": label_color,
                "letterSpacing": "0.1em", "fontWeight": "700"}),
            html.H3(title_parts, style={"margin": "8px 0 12px 0", "fontSize": "1.2em"}),
            html.P(body, style={"color": "#334155", "lineHeight": "1.65", "margin": 0}),
        ])

app.layout = html.Div(
    style={"fontFamily": "'Inter', system-ui, -apple-system, sans-serif",
           "background": "#fafafa", "minHeight": "100vh", "color": TEXT},
    children=[html.Div(style={"maxWidth": "1200px", "margin": "0 auto", "padding": "48px 24px"},
        children=[
            # HERO
            html.Div([
                html.Div("2026 FIFA WORLD CUP", style={"fontSize": "0.85em", "color": MUTED,
                    "letterSpacing": "0.15em", "fontWeight": "600"}),
                html.H1("Machine learning vs. football intuition",
                    style={"fontSize": "2.6em", "fontWeight": "700", "margin": "8px 0",
                        "letterSpacing": "-0.02em", "lineHeight": "1.1"}),
                html.P(["Two predictions, both locked before the first match was played. ",
                    html.A("Code and methodology on GitHub →",
                        href="https://github.com/abhimanyududeja/worldcup26-predictions",
                        target="_blank",
                        style={"color": ML_C, "textDecoration": "none", "fontWeight": "500"})],
                    style={"fontSize": "1.1em", "color": MUTED, "marginTop": "8px"}),
                html.Div([
                    html.Span("LOCKED", style={"background": "#dcfce7", "color": "#166534",
                        "padding": "3px 10px", "borderRadius": "4px", "fontSize": "0.7em",
                        "fontWeight": "700", "letterSpacing": "0.05em"}),
                    html.Span(f"  {locked['locked_at_utc']}", style={"color": MUTED,
                        "fontSize": "0.85em", "marginLeft": "8px", "fontFamily": "monospace"}),
                ], style={"marginTop": "20px"}),
            ], style={"marginBottom": "48px"}),

            # HEADLINE CARDS
            
            # Hero TL;DR - the whole thesis in one banner
            html.Div(style={"background": "white", "padding": "28px 32px",
                "borderRadius": "14px", "border": f"2px solid {ML_C}",
                "marginBottom": "32px", "fontSize": "1.1em", "lineHeight": "1.7"},
                children=[
                    html.Div("THE BET", style={"fontSize": "0.72em", "color": ML_C,
                        "letterSpacing": "0.15em", "fontWeight": "700",
                        "marginBottom": "14px"}),
                    html.Div([
                        html.Span("Three predictions, all locked before kickoff. "),
                        html.Strong("My ML model says Spain wins. ",
                            style={"color": ML_C}),
                        html.Strong("I say France wins. ",
                            style={"color": FAN_C}),
                        html.Strong("FanDuel has them co-favorites. ",
                            style={"color": "#7c3aed"}),
                        "Two of these three will be more wrong than the third. ",
                        html.Span("Scroll to see where they disagree.",
                            style={"color": MUTED, "fontStyle": "italic"}),
                    ]),
                ]),

            html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                children=[
                    champion_card("ML MODEL PREDICTS", ml_bracket["champion"],
                        f"{ml_probs[0]['p_win']*100:.1f}% to win the tournament", ML_C, "#eff6ff"),
                    champion_card("FAN BRACKET", fan_bracket["champion"],
                        "Picked by Abhimanyu, a football fan. No model involved.", FAN_C, "#fffbeb"),
                ]),

            # TOP 10 ML
            section_title("ML model: top 10 most likely champions",
                "From 10,000 Monte Carlo tournament simulations"),
            html.Div(style={"background": "white", "padding": "24px", "borderRadius": "14px",
                "border": f"1px solid {BORDER}"},
                children=[dcc.Graph(figure=fig_top10, config={"displayModeBar": False})]),

            # FAN BRACKET CHART
            section_title("Fan bracket: how deep each team goes",
                "My picks visualized. Eight teams reach the quarter-finals, narrowing to a France final win."),
            html.Div(style={"background": "white", "padding": "24px", "borderRadius": "14px",
                "border": f"1px solid {BORDER}"},
                children=[dcc.Graph(figure=fig_fan, config={"displayModeBar": False})]),

            # DIVERGENCE TABLE
            section_title("Where my football brain disagrees with the model",
                "Side-by-side picks for each knockout stage."),
            html.Div(style={"background": "white", "borderRadius": "14px",
                "border": f"1px solid {BORDER}", "overflow": "hidden"},
                children=[html.Table(style={"width": "100%", "borderCollapse": "collapse"},
                    children=[
                        html.Thead(html.Tr([
                            html.Th("STAGE", style={"textAlign": "left", "padding": "14px 20px",
                                "background": BG_LIGHT, "fontSize": "0.72em", "letterSpacing": "0.08em",
                                "color": MUTED, "fontWeight": "700"}),
                            html.Th("ML MODEL", style={"textAlign": "left", "padding": "14px 20px",
                                "background": BG_LIGHT, "fontSize": "0.72em", "letterSpacing": "0.08em",
                                "color": ML_C, "fontWeight": "700"}),
                            html.Th("FAN BRACKET", style={"textAlign": "left", "padding": "14px 20px",
                                "background": BG_LIGHT, "fontSize": "0.72em", "letterSpacing": "0.08em",
                                "color": FAN_C, "fontWeight": "700"}),
                        ])),
                        html.Tbody([
                            stage_row("Champion",
                                [fl(ml_bracket["champion"]), " ", html.Strong(ml_bracket["champion"])],
                                [fl(fan_bracket["champion"]), " ", html.Strong(fan_bracket["champion"])]),
                            stage_row("Finalists", team_list(ml_bracket["finalists"]),
                                team_list(fan_bracket["finalists"])),
                            stage_row("Semi-finals", team_list(sorted(ml_bracket["sf_teams"])),
                                team_list(sorted(fan_bracket["sf_teams"]))),
                            stage_row("Quarter-finals", team_list(sorted(ml_bracket["qf_teams"])),
                                team_list(sorted(fan_bracket["qf_teams"]))),
                        ])
                    ])
                ]),

            # WHERE I THINK THE MODEL IS WRONG
            section_title("My reasoning for picking France over Spain",
                "Where I disagree with the model, and why."),
            html.Div(style={"background": "white", "padding": "32px", "borderRadius": "12px",
                "border": f"1px solid {BORDER}", "borderLeft": f"4px solid {FAN_C}"},
                children=[
                    html.Div("WHY I PICKED FRANCE", style={"fontSize": "0.7em", "color": FAN_C,
                        "letterSpacing": "0.1em", "fontWeight": "700"}),
                    html.H3([fl("France"), " France over ", fl("Spain"), " Spain"],
                        style={"margin": "8px 0 16px 0", "fontSize": "1.4em"}),
                    html.P([
                        "Both Spain and France are the top tier of championship contenders. ",
                        "The model has Spain at #1 (21.6%), but my read is they are much closer than that. ",
                        "The 2026 bracket positioning likely puts them on a collision course in the round of 16 or 32. Only one of them can make a deep run."
                    ], style={"color": "#334155", "lineHeight": "1.7", "marginBottom": "12px"}),
                    html.P([
                        html.Strong("I picked France for that head-to-head. "),
                        "They have made the last two World Cup finals (winning 2018, runner-up 2022). ",
                        "Their squad value is the highest in the tournament (€1.53B vs Spain at €1.26B). ",
                        "Spain is the reigning Euro champion, but France has the World Cup tournament experience that I trust more when it counts."
                    ], style={"color": "#334155", "lineHeight": "1.7", "margin": 0}),
                ]),

            section_title("Backtest: model performance on past World Cups",
                "Same Elo + Poisson core, retrained on pre-tournament data only (no lookahead bias). Squad value adjustment not included (no historical squad value data)."),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
                children=[make_backtest_card(backtest["2022"]), make_backtest_card(backtest["2018"])]),
            html.Div(style={"marginTop": "20px", "padding": "24px", "background": BG_LIGHT, "borderRadius": "10px", "border": f"1px solid {BORDER}"},
                children=[
                    html.H4("What the backtest reveals", style={"margin": "0 0 8px 0"}),
                    html.P(["On 64 actual matches from each tournament, the model beats the uniform baseline by ",
                        html.Strong("6.6% in 2022 and 11.0% in 2018"),
                        ". It identified Argentina as #2 in 2022 (Argentina won) and France as #4 in 2018 (France won). Both actual finalists from both tournaments landed in the top 5."],
                        style={"color": "#334155", "lineHeight": "1.7"}),
                    html.P([html.Strong("The CONMEBOL inflation problem is real. "),
                        "Brazil was the model's #1 pick in BOTH 2018 and 2022, with 25%+ win probability each time. Brazil won neither. This is exactly the pattern called out in the \"where the model is wrong\" section above. The 2026 model's squad value adjustment partially corrects this: Brazil drops from a hypothetical #1 (pure Elo) to #5 (6.3% in 2026)."],
                        style={"color": "#334155", "lineHeight": "1.7"}),
                    html.P([html.Strong("Honest miss: "), "Croatia's run to the 2018 final (model ranked them #13). Almost no pre-tournament model predicted that run, but it's a real failure mode worth surfacing."],
                        style={"color": "#334155", "lineHeight": "1.7", "margin": 0}),
                ]),

                        section_title("Calibration: are the probabilities meaningful?",
                f"When the model says 70%, does the event happen 70% of the time? Computed from {calibration['n_total']} prediction events (home win / draw / away win) across the 2018 and 2022 backtests."),
            html.Div(style={"background": "white", "padding": "24px", "borderRadius": "14px", "border": f"1px solid {BORDER}"},
                children=[dcc.Graph(figure=fig_calib, config={"displayModeBar": False})]),
            html.Div(style={"marginTop": "20px", "padding": "24px", "background": BG_LIGHT, "borderRadius": "10px", "border": f"1px solid {BORDER}"},
                children=[
                    html.H4("Reading the chart", style={"margin": "0 0 8px 0"}),
                    html.P([
                        "Each point is a probability bin (e.g., 20-30%). The X-axis is the model's average predicted probability in that bin. The Y-axis is how often the predicted outcome actually happened. ",
                        html.Strong("If the model is well-calibrated, points fall on the dashed line. "),
                        "Marker size reflects the number of predictions in the bin."
                    ], style={"color": "#334155", "lineHeight": "1.7"}),
                    html.P([
                        html.Strong(f"Expected Calibration Error: {calibration['expected_calibration_error']*100:.2f}%"),
                        " across all 384 prediction events. The model is well-calibrated in the 10-50% range (the bins with the most samples). It is slightly underconfident in the 50-70% range, predicting 55-65% when the outcome actually happens 60-70% of the time. The extreme bins (0-10%, 80-90%) have too few samples to draw firm conclusions."
                    ], style={"color": "#334155", "lineHeight": "1.7", "margin": 0}),
                ]),

                        section_title("Model vs. market",
                f"How my ML model compares to current FanDuel sportsbook odds, with the bookmaker's vig ({(bookmaker['overround']-1)*100:.1f}%) removed. Snapshot from {bookmaker['as_of']}."),
            html.Div(style={"background": "white", "padding": "24px", "borderRadius": "14px", "border": f"1px solid {BORDER}"},
                children=[dcc.Graph(figure=fig_market, config={"displayModeBar": False})]),
            html.Div(style={"display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap": "16px", "marginTop": "20px"}, children=[

                html.Div(style={"background": "white", "padding": "24px",
                    "borderRadius": "12px", "border": f"1px solid {BORDER}",
                    "borderTop": f"4px solid {FAN_C}"}, children=[
                    html.Div("PORTUGAL", style={"fontSize": "0.72em", "color": FAN_C,
                        "letterSpacing": "0.12em", "fontWeight": "700"}),
                    html.H3("Market validates my call",
                        style={"margin": "8px 0 16px 0", "fontSize": "1.15em"}),
                    html.Div(style={"display": "flex", "gap": "24px", "marginBottom": "12px"}, children=[
                        html.Div([html.Div("MARKET", style={"fontSize": "0.7em", "color": MUTED}),
                                  html.Div("8.2%", style={"fontSize": "1.6em", "fontWeight": "700"})]),
                        html.Div([html.Div("MODEL", style={"fontSize": "0.7em", "color": MUTED}),
                                  html.Div("3.5%", style={"fontSize": "1.6em", "fontWeight": "700",
                                  "color": MUTED})]),
                    ]),
                    html.Div("Sharp bettors price Portugal as a top-5 contender. Model has them 8th.",
                        style={"color": "#334155", "fontSize": "0.92em", "lineHeight": "1.5"}),
                ]),

                html.Div(style={"background": "white", "padding": "24px",
                    "borderRadius": "12px", "border": f"1px solid {BORDER}",
                    "borderTop": "4px solid #dc2626"}, children=[
                    html.Div("CONMEBOL INFLATION", style={"fontSize": "0.72em", "color": "#dc2626",
                        "letterSpacing": "0.12em", "fontWeight": "700"}),
                    html.H3("Confirmed by the market",
                        style={"margin": "8px 0 16px 0", "fontSize": "1.15em"}),
                    html.Div(style={"display": "flex", "gap": "24px", "marginBottom": "12px", "flexWrap": "wrap"}, children=[
                        html.Div([html.Div("ECUADOR", style={"fontSize": "0.7em", "color": MUTED}),
                                  html.Div("4.4% vs 1.1%", style={"fontSize": "1.1em", "fontWeight": "700"})]),
                        html.Div([html.Div("COLOMBIA", style={"fontSize": "0.7em", "color": MUTED}),
                                  html.Div("4.5% vs 2.2%", style={"fontSize": "1.1em", "fontWeight": "700"})]),
                    ]),
                    html.Div("Same pattern visible in 2018 and 2022 backtests. The market does not share the model's confidence in South American teams.",
                        style={"color": "#334155", "fontSize": "0.92em", "lineHeight": "1.5"}),
                ]),

                html.Div(style={"background": "white", "padding": "24px",
                    "borderRadius": "12px", "border": f"1px solid {BORDER}",
                    "borderTop": f"4px solid {ML_C}"}, children=[
                    html.Div("SPAIN VS FRANCE", style={"fontSize": "0.72em", "color": ML_C,
                        "letterSpacing": "0.12em", "fontWeight": "700"}),
                    html.H3("Market sides with my gut",
                        style={"margin": "8px 0 16px 0", "fontSize": "1.15em"}),
                    html.Div(style={"display": "flex", "gap": "24px", "marginBottom": "12px", "flexWrap": "wrap"}, children=[
                        html.Div([html.Div("MARKET", style={"fontSize": "0.7em", "color": MUTED}),
                                  html.Div("15.8 / 15.1%", style={"fontSize": "1.1em", "fontWeight": "700"}),
                                  html.Div("Tied", style={"fontSize": "0.78em", "color": MUTED})]),
                        html.Div([html.Div("MODEL", style={"fontSize": "0.7em", "color": MUTED}),
                                  html.Div("21.6 / 10.7%", style={"fontSize": "1.1em", "fontWeight": "700"}),
                                  html.Div("11pt gap", style={"fontSize": "0.78em", "color": MUTED})]),
                    ]),
                    html.Div("Market and fan bracket both pick France over Spain. Model strongly disagrees.",
                        style={"color": "#334155", "fontSize": "0.92em", "lineHeight": "1.5"}),
                ]),
            ]),

            section_title("How the model rates every team in every group",
                "Teams ranked by adjusted Elo. 'Adv' = model's predicted probability of advancing to R32."),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
                children=[make_group_card(g) for g in sorted(group_teams.keys())]),

            # HARDEST CALLS
            section_title("Coin flips: the matches the model is least sure about",
                "Group matches with the highest prediction entropy - closest to 50/50."),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
                children=uncertain_cards),

            # 72 MATCHES
            section_title("Every group match with a locked prediction",
                "All 72 matches at neutral venues. 'Win (1st)' refers to the team listed first in the match; 'Win (2nd)' to the team listed second. xG = expected goals (Poisson rate parameter)."),
            dash_table.DataTable(data=match_table,
                columns=[{"name": "Date", "id": "Date"}, {"name": "Grp", "id": "G"},
                    {"name": "Match", "id": "Match"}, {"name": "Win (1st)", "id": "Home"},
                    {"name": "Draw", "id": "Draw"}, {"name": "Win (2nd)", "id": "Away"},
                    {"name": "xG (1st)", "id": "xG_H"}, {"name": "xG (2nd)", "id": "xG_A"},
                    {"name": "Most Likely", "id": "Likely"}],
                sort_action="native", page_size=18,
                style_table={"borderRadius": "10px", "overflow": "hidden", "border": f"1px solid {BORDER}"},
                style_cell={"textAlign": "left", "padding": "10px 12px",
                    "fontFamily": "'Inter', system-ui, sans-serif", "fontSize": "0.92em",
                    "border": "none", "borderBottom": f"1px solid {BORDER}"},
                style_header={"backgroundColor": BG_LIGHT, "color": MUTED, "fontWeight": "700",
                    "fontSize": "0.72em", "letterSpacing": "0.08em", "textTransform": "uppercase",
                    "padding": "12px"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fdfdfd"}],
            ),

            # 48 TEAMS
            section_title("Every team's odds at every stage",
                "Each cell = probability the team reaches that stage or beyond, from 10,000 simulations."),
            dash_table.DataTable(
                data=[{"T": f"{fl(r['team'])} {r['team']}", "Elo": int(r["elo_adj"]),
                    "R16": f"{r['p_r16']*100:.1f}%", "QF": f"{r['p_qf']*100:.1f}%",
                    "SF": f"{r['p_sf']*100:.1f}%", "F": f"{r['p_final']*100:.1f}%",
                    "W": f"{r['p_win']*100:.1f}%"} for r in ml_probs],
                columns=[{"name": "Team", "id": "T"}, {"name": "Adj. Elo", "id": "Elo"},
                    {"name": "Reach R16", "id": "R16"}, {"name": "Reach QF", "id": "QF"},
                    {"name": "Reach SF", "id": "SF"}, {"name": "Reach Final", "id": "F"},
                    {"name": "Win", "id": "W"}],
                sort_action="native", page_size=15,
                style_table={"borderRadius": "10px", "overflow": "hidden", "border": f"1px solid {BORDER}"},
                style_cell={"textAlign": "left", "padding": "10px 12px",
                    "fontFamily": "'Inter', system-ui, sans-serif", "fontSize": "0.92em",
                    "border": "none", "borderBottom": f"1px solid {BORDER}"},
                style_header={"backgroundColor": BG_LIGHT, "color": MUTED, "fontWeight": "700",
                    "fontSize": "0.72em", "letterSpacing": "0.08em", "textTransform": "uppercase",
                    "padding": "12px"},
            ),

            # METHODOLOGY
            section_title("Methodology"),
            html.Div(style={"display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "16px"},
                children=[
                    html.Div(style={"background": "white", "padding": "20px", "borderRadius": "10px",
                        "border": f"1px solid {BORDER}"},
                        children=[
                            html.Div("HISTORICAL DATA", style={"fontSize": "0.7em", "color": MUTED,
                                "letterSpacing": "0.1em", "fontWeight": "700"}),
                            html.Div("49,378", style={"fontSize": "2em", "fontWeight": "700", "margin": "8px 0"}),
                            html.Div("International matches, 1872–2026", style={"color": MUTED, "fontSize": "0.9em"})]),
                    html.Div(style={"background": "white", "padding": "20px", "borderRadius": "10px",
                        "border": f"1px solid {BORDER}"},
                        children=[
                            html.Div("MODEL TEST LOG LOSS", style={"fontSize": "0.7em", "color": MUTED,
                                "letterSpacing": "0.1em", "fontWeight": "700"}),
                            html.Div("0.887", style={"fontSize": "2em", "fontWeight": "700", "margin": "8px 0"}),
                            html.Div("vs. baseline 1.043 - beats baseline by 15%",
                                style={"color": MUTED, "fontSize": "0.9em"})]),
                    html.Div(style={"background": "white", "padding": "20px", "borderRadius": "10px",
                        "border": f"1px solid {BORDER}"},
                        children=[
                            html.Div("SIMULATIONS", style={"fontSize": "0.7em", "color": MUTED,
                                "letterSpacing": "0.1em", "fontWeight": "700"}),
                            html.Div("10,000", style={"fontSize": "2em", "fontWeight": "700", "margin": "8px 0"}),
                            html.Div("Monte Carlo runs of the full bracket",
                                style={"color": MUTED, "fontSize": "0.9em"})]),
                ]),

            html.Div(style={"marginTop": "20px", "padding": "28px", "background": "white",
                "borderRadius": "10px", "border": f"1px solid {BORDER}"},
                children=[
                    html.H4("Two signals", style={"margin": "0 0 8px 0"}),
                    html.P([html.Strong("Elo ratings"), " over 49,378 historical international matches, "
                        "with K-factor varying by tournament importance (friendlies K=20, World Cup K=60), "
                        "home advantage +65 Elo, and goal-difference scaling. Combined with a ",
                        html.Strong("bivariate Poisson score model"),
                        " for goal counts (log loss 0.887 vs. 1.043 baseline)."],
                        style={"color": "#334155", "lineHeight": "1.7"}),
                    html.H4("Squad value adjustment", style={"margin": "20px 0 8px 0"}),
                    html.P(["Elo alone underrates teams whose squads are stronger than their recent results "
                        "and overrates teams whose qualifying campaigns inflated their Elo. The model z-scores log market value "
                        "across 48 teams and adds 75 Elo per standard deviation."],
                        style={"color": "#334155", "lineHeight": "1.7"}),
                    html.H4("Known limitations", style={"margin": "20px 0 8px 0"}),
                    html.Ul([
                        html.Li("Simplified knockout bracket - Elo-seeded pairing rather than the actual 2026 slot assignments."),
                        html.Li("No player-level features - injuries, form, fitness all missing."),
                        html.Li("No aggressive recency weighting - 2018 results still influence ratings."),
                        html.Li("CONMEBOL Elo inflation partially corrected by squad value, not fully (see above)."),
                        html.Li("Penalty shootouts modeled as Elo-weighted coin flips."),
                    ], style={"color": "#334155", "lineHeight": "1.8"}),
                ]),

            html.Hr(style={"marginTop": "56px", "border": "none", "borderTop": f"1px solid {BORDER}"}),
            html.Div(style={"textAlign": "center", "color": MUTED, "fontSize": "0.85em",
                "marginTop": "24px", "paddingBottom": "40px"},
                children=[html.Div(["Built by Abhimanyu Dudeja · ",
                    html.A("GitHub", href="https://github.com/abhimanyududeja/worldcup26-predictions",
                        target="_blank", style={"color": ML_C, "textDecoration": "none"}),
                    " · Locked before kickoff, scored against reality as the tournament unfolds."])]),
        ])]
)


app.index_string = """<!DOCTYPE html>
<html>
<head>
<title>{%title%}</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%231e3a8a'/><text x='50' y='68' font-size='48' text-anchor='middle' fill='white' font-family='Inter,system-ui,sans-serif' font-weight='700'>26</text></svg>">
{%metas%}
{%css%}
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
