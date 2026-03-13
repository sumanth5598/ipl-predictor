import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path

st.set_page_config(page_title="IPL Predictor", page_icon="IPL", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0a0a0f;color:#f0f0f0}
.main-title{font-family:'Bebas Neue',sans-serif;font-size:3rem;letter-spacing:4px;
background:linear-gradient(135deg,#f5a623,#e8320a);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;text-align:center}
.subtitle{text-align:center;color:#888;font-size:0.85rem;letter-spacing:3px;
text-transform:uppercase;margin-bottom:1.5rem}
.lbl{font-size:0.7rem;letter-spacing:3px;text-transform:uppercase;color:#666;margin-bottom:0.2rem}
.pred-win{background:linear-gradient(135deg,#1a3a1a,#0d2b0d);border:1px solid #2d7a2d;
border-radius:16px;padding:1.5rem;text-align:center;margin:0.6rem 0}
.pred-lose{background:linear-gradient(135deg,#3a1a1a,#2b0d0d);border:1px solid #7a2d2d;
border-radius:16px;padding:1.5rem;text-align:center;margin:0.6rem 0}
.pred-team{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:2px}
.pred-pct{font-size:2.5rem;font-weight:600;margin:0.3rem 0}
.win-pct{color:#4caf50}.lose-pct{color:#ef5350}
.score-box{background:linear-gradient(135deg,#1a1a3a,#0d0d2b);border:1px solid #3a3a7a;
border-radius:16px;padding:1.5rem;text-align:center;margin:0.6rem 0}
.score-num{font-family:'Bebas Neue',sans-serif;font-size:3.5rem;letter-spacing:3px;
background:linear-gradient(135deg,#f5a623,#e8320a);-webkit-background-clip:text;
-webkit-text-fill-color:transparent}
.over-box{background:linear-gradient(135deg,#1a3a1a,#0d2b0d);border:2px solid #4caf50;
border-radius:16px;padding:1.5rem;text-align:center;margin:0.6rem 0}
.under-box{background:linear-gradient(135deg,#3a1a1a,#2b0d0d);border:2px solid #ef5350;
border-radius:16px;padding:1.5rem;text-align:center;margin:0.6rem 0}
.verdict-text{font-family:"Bebas Neue",sans-serif;font-size:3rem;letter-spacing:4px}
.over-text{color:#4caf50}.under-text{color:#ef5350}
.verdict-sub{font-size:0.88rem;color:#aaa;margin-top:0.3rem}
.prob-bar-bg{background:#1e1e2e;border-radius:50px;height:12px;overflow:hidden;margin:1rem 0}
.prob-bar-fill{height:100%;border-radius:50px;background:linear-gradient(90deg,#e8320a,#f5a623)}
.chip{display:inline-block;background:#1e1e2e;border:1px solid #333;border-radius:8px;
padding:0.3rem 0.7rem;font-size:0.75rem;color:#aaa;margin:0.2rem}
.badge{background:#1a1a2e;border:1px solid #f5a623;border-radius:8px;padding:0.4rem 1rem;
text-align:center;font-size:0.75rem;color:#f5a623;letter-spacing:1px}
.divider{border:none;border-top:1px solid #222;margin:1.2rem 0}
.stage-header{font-family:"Bebas Neue",sans-serif;font-size:1.3rem;letter-spacing:2px;
color:#f5a623;margin:1rem 0 0.5rem 0}
div[data-testid="stSelectbox"]>div{background:#1e1e2e!important;border:1px solid #333!important;border-radius:10px!important}
.stButton>button{background:linear-gradient(135deg,#e8320a,#f5a623)!important;color:white!important;
font-family:"Bebas Neue",sans-serif!important;font-size:1.2rem!important;letter-spacing:3px!important;
border:none!important;border-radius:12px!important;padding:0.7rem 2rem!important;width:100%!important}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent

@st.cache_resource
def load_all():
    model    = pickle.load(open(BASE/"best_ipl_predictor.pkl",  "rb"))
    le_team  = pickle.load(open(BASE/"label_encoder_team.pkl",  "rb"))
    le_venue = pickle.load(open(BASE/"label_encoder_venue.pkl", "rb"))
    features = json.load(open(BASE/"features_final.json"))
    medians  = json.load(open(BASE/"train_medians.json"))
    caps_raw = json.load(open(BASE/"captains.json"))
    captains = {eval(k): v for k, v in caps_raw.items()}
    mA       = pickle.load(open(BASE/"score_model_toss_to_6.pkl",  "rb"))
    mB       = pickle.load(open(BASE/"score_model_6_to_10.pkl",    "rb"))
    mC       = pickle.load(open(BASE/"score_model_10_to_final.pkl","rb"))
    sf       = json.load(open(BASE/"score_features_v2.json"))
    sm       = json.load(open(BASE/"score_medians_v2.json"))
    return model,le_team,le_venue,features,medians,captains,mA,mB,mC,sf,sm

model,le_team,le_venue,features,medians,captains,mA,mB,mC,sf,sm = load_all()

TEAMS = sorted(["Mumbai Indians","Chennai Super Kings","Kolkata Knight Riders",
    "Royal Challengers Bengaluru","Delhi Capitals","Rajasthan Royals",
    "Punjab Kings","Sunrisers Hyderabad","Gujarat Titans","Lucknow Super Giants"])

VENUES = sorted(["Wankhede Stadium","M Chinnaswamy Stadium","Eden Gardens",
    "MA Chidambaram Stadium","Arun Jaitley Stadium",
    "Rajiv Gandhi International Cricket Stadium","Narendra Modi Stadium",
    "Sawai Mansingh Stadium","Punjab Cricket Association IS Bindra Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Dr DY Patil Sports Academy","Brabourne Stadium"])

CITIES = {"Wankhede Stadium":"Mumbai","M Chinnaswamy Stadium":"Bangalore",
    "Eden Gardens":"Kolkata","MA Chidambaram Stadium":"Chennai",
    "Arun Jaitley Stadium":"Delhi",
    "Rajiv Gandhi International Cricket Stadium":"Hyderabad",
    "Narendra Modi Stadium":"Ahmedabad","Sawai Mansingh Stadium":"Jaipur",
    "Punjab Cricket Association IS Bindra Stadium":"Chandigarh",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium":"Lucknow",
    "Dr DY Patil Sports Academy":"Mumbai","Brabourne Stadium":"Mumbai"}

HOME_CITY = {"Mumbai Indians":"Mumbai","Chennai Super Kings":"Chennai",
    "Kolkata Knight Riders":"Kolkata","Royal Challengers Bengaluru":"Bangalore",
    "Delhi Capitals":"Delhi","Rajasthan Royals":"Jaipur",
    "Punjab Kings":"Chandigarh","Sunrisers Hyderabad":"Hyderabad",
    "Gujarat Titans":"Ahmedabad","Lucknow Super Giants":"Lucknow"}

def enc(le, val, fb=0):
    try: return int(le.transform([val])[0])
    except: return fb

def predict_winner(team1, team2, venue, season, toss_w, toss_d):
    city = CITIES.get(venue,"")
    row  = {k: medians.get(k,0.5) for k in features}
    row.update({
        "team1_encoded"       : enc(le_team,  team1),
        "team2_encoded"       : enc(le_team,  team2),
        "venue_encoded"       : enc(le_venue, venue),
        "season"              : season,
        "toss_winner_is_team1": 1 if toss_w==team1 else 0,
        "toss_bat_first"      : 1 if ((toss_w==team1 and toss_d=="Bat") or
                                       (toss_w==team2 and toss_d=="Field")) else 0,
        "team1_is_home": 1 if HOME_CITY.get(team1,"") in city else 0,
        "team2_is_home": 1 if HOME_CITY.get(team2,"") in city else 0,
    })
    prob = model.predict_proba(pd.DataFrame([row])[features])[0]
    c1 = captains.get((team1,season), captains.get((team1,2024),"Unknown"))
    c2 = captains.get((team2,season), captains.get((team2,2024),"Unknown"))
    return float(prob[1]), float(prob[0]), c1, c2

def stage_A(bat_team, venue, season):
    v = enc(le_venue, venue, 0)
    t = enc(le_team,  bat_team, 0)
    th = sm.get("team_hist_avg", 160)
    vh = sm.get("venue_hist_avg", 160)
    row = {"venue_enc":v,"team_enc":t,"season":season,
           "team_hist_avg":th,"venue_hist_avg":vh}
    pred = mA.predict(pd.DataFrame([row])[sf["fA"]])[0]
    return round(float(pred)), 11

def stage_B(bat_team, venue, season, pp_runs, pp_wkts):
    v     = enc(le_venue, venue, 0)
    t     = enc(le_team,  bat_team, 0)
    pp_rr = pp_runs / 36 * 6
    pp_br = sm.get("pp_boundary_rate", 0.15)
    pp_dr = sm.get("pp_dot_rate", 0.35)
    pp_wr = 10 - pp_wkts
    th    = sm.get("team_hist_avg", 160)
    vh    = sm.get("venue_hist_avg", 160)
    row   = {"pp_runs":pp_runs,"pp_wickets":pp_wkts,"pp_run_rate":pp_rr,
             "pp_boundary_rate":pp_br,"pp_dot_rate":pp_dr,
             "pp_wkts_remaining":pp_wr,
             "venue_enc":v,"team_enc":t,"season":season,
             "team_hist_avg":th,"venue_hist_avg":vh}
    pred = mB.predict(pd.DataFrame([row])[sf["fB"]])[0]
    return round(float(pred)), 9

def stage_C(bat_team, venue, season,
            pp_runs, pp_wkts, mid_runs, mid_wkts):
    v      = enc(le_venue, venue, 0)
    t      = enc(le_team,  bat_team, 0)
    pp_rr  = pp_runs / 36 * 6
    mid_rr = mid_runs / 60 * 6
    mid_or = mid_runs - pp_runs
    mid_ac = mid_or / (pp_runs + 1)
    row = {k: sm.get(k,0) for k in sf["fC"]}
    row.update({"pp_runs":pp_runs,"pp_wickets":pp_wkts,"pp_run_rate":pp_rr,
                "pp_boundary_rate":sm.get("pp_boundary_rate",0.15),
                "pp_dot_rate":sm.get("pp_dot_rate",0.35),
                "pp_wkts_remaining":10-pp_wkts,
                "mid_runs":mid_runs,"mid_wickets":mid_wkts,"mid_run_rate":mid_rr,
                "mid_boundary_rate":sm.get("mid_boundary_rate",0.15),
                "mid_dot_rate":sm.get("mid_dot_rate",0.35),
                "mid_wkts_remaining":10-mid_wkts,
                "mid_only_runs":mid_or,"mid_acceleration":mid_ac,
                "venue_enc":v,"team_enc":t,"season":season,
                "team_hist_avg":sm.get("team_hist_avg",160),
                "venue_hist_avg":sm.get("venue_hist_avg",160)})
    pred = mC.predict(pd.DataFrame([row])[sf["fC"]])[0]
    return round(float(pred)), 20

def show_over_under(pred, mae, user_target, label):
    lo = max(pred-mae, 80)
    hi = pred+mae
    if pred > user_target:
        box  = "over-box"
        vt   = "over-text"
        txt  = "OVER"
        desc = f"Model predicts {pred} — {pred-user_target} above your target of {user_target}"
    elif pred < user_target:
        box  = "under-box"
        vt   = "under-text"
        txt  = "UNDER"
        desc = f"Model predicts {pred} — {user_target-pred} below your target of {user_target}"
    else:
        box  = "over-box"
        vt   = "over-text"
        txt  = "EXACT"
        desc = f"Model predicts exactly {pred} — matches your target!"
    st.markdown(f"""
    <div class="{box}">
        <div class="lbl">OVER / UNDER {user_target} — {label}</div>
        <div class="verdict-text {vt}">{txt}</div>
        <div class="verdict-sub">{desc}</div>
    </div>
    <div class="score-box">
        <div class="lbl">PREDICTED SCORE</div>
        <div class="score-num">{pred}</div>
        <div style="color:#888;font-size:0.82rem">Range: {int(lo)}-{int(hi)} | MAE +/-{mae} runs</div>
    </div>""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("<p class='main-title'>IPL PREDICTOR</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Machine Learning · 2008-2025 · XGBoost</p>", unsafe_allow_html=True)
st.markdown("<div class='badge'>Win: 68-76% accuracy | Score MAE: 6ov +/-11 | 10ov +/-9 | Final +/-20 runs</div>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["MATCH WINNER", "SCORE PREDICTOR"])

# ══ TAB 1 — Match Winner ══════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<p class='lbl'>Team 1</p>", unsafe_allow_html=True)
        team1 = st.selectbox("", TEAMS, index=0, key="t1", label_visibility="collapsed")
    with c2:
        st.markdown("<p class='lbl'>Team 2</p>", unsafe_allow_html=True)
        team2 = st.selectbox("", TEAMS, index=1, key="t2", label_visibility="collapsed")
    c3,c4 = st.columns([2,1])
    with c3:
        st.markdown("<p class='lbl'>Venue</p>", unsafe_allow_html=True)
        venue = st.selectbox("", VENUES, key="v1", label_visibility="collapsed")
    with c4:
        st.markdown("<p class='lbl'>Season</p>", unsafe_allow_html=True)
        season = st.selectbox("", list(range(2026,2007,-1)), key="s1", label_visibility="collapsed")
    c5,c6 = st.columns(2)
    with c5:
        st.markdown("<p class='lbl'>Toss Won By</p>", unsafe_allow_html=True)
        toss_w = st.selectbox("", [team1,team2], key="tw", label_visibility="collapsed")
    with c6:
        st.markdown("<p class='lbl'>Chose To</p>", unsafe_allow_html=True)
        toss_d = st.selectbox("", ["Bat","Field"], key="td", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("PREDICT WINNER", key="btn_win"):
        if team1==team2:
            st.error("Please select two different teams!")
        else:
            p1,p2,cap1,cap2 = predict_winner(team1,team2,venue,season,toss_w,toss_d)
            winner = team1 if p1>=p2 else team2
            loser  = team2 if p1>=p2 else team1
            wp,lp  = max(p1,p2), min(p1,p2)
            st.markdown(f"""
            <div class="pred-win">
                <div class="lbl">PREDICTED WINNER</div>
                <div class="pred-team">{winner}</div>
                <div class="pred-pct win-pct">{wp*100:.1f}%</div>
            </div>
            <div class="pred-lose">
                <div class="lbl">RUNNER UP</div>
                <div class="pred-team">{loser}</div>
                <div class="pred-pct lose-pct">{lp*100:.1f}%</div>
            </div>
            <p class="lbl" style="margin-top:1rem">WIN PROBABILITY — {team1}</p>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p1*100:.1f}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#888">
                <span>{team1} {p1*100:.1f}%</span><span>{p2*100:.1f}% {team2}</span>
            </div>
            <hr class="divider">
            <div>
                <span class="chip">{CITIES.get(venue,"Neutral")}</span>
                <span class="chip">{team1}: {cap1}</span>
                <span class="chip">{team2}: {cap2}</span>
                <span class="chip">{toss_w} chose {toss_d.lower()}</span>
            </div>""", unsafe_allow_html=True)

# ══ TAB 2 — Score Predictor ═══════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<p class='lbl'>Batting Team</p>", unsafe_allow_html=True)
        bat_team = st.selectbox("", TEAMS, key="bt", label_visibility="collapsed")
    with c2:
        st.markdown("<p class='lbl'>Venue</p>", unsafe_allow_html=True)
        bat_venue = st.selectbox("", VENUES, key="bv", label_visibility="collapsed")
    c3,c4 = st.columns([1,2])
    with c3:
        st.markdown("<p class='lbl'>Season</p>", unsafe_allow_html=True)
        bat_season = st.selectbox("", list(range(2026,2007,-1)), key="bs", label_visibility="collapsed")
    with c4:
        st.markdown("<p class='lbl'>Stage</p>", unsafe_allow_html=True)
        stage = st.selectbox("",
            ["Stage 1 — After Toss (predict 6 over score)",
             "Stage 2 — After 6 Overs (predict 10 over score)",
             "Stage 3 — After 10 Overs (predict final score + Over/Under)"],
            key="stg", label_visibility="collapsed")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    pp_runs=pp_wkts=mid_runs=mid_wkts=None

    if "Stage 1" in stage:
        st.markdown("<p class='stage-header'>STAGE 1 — AFTER TOSS</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#888;font-size:0.82rem'>Predicts expected 6-over powerplay score based on team and venue history</p>", unsafe_allow_html=True)

    if "Stage 2" in stage:
        st.markdown("<p class='stage-header'>STAGE 2 — AFTER 6 OVERS</p>", unsafe_allow_html=True)
        ca,cb = st.columns(2)
        with ca: pp_runs = st.number_input("PP Runs (0-6 ov)",  0, 120, 50, key="pr")
        with cb: pp_wkts = st.number_input("PP Wickets (0-6 ov)", 0, 10, 1, key="pw")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#888;font-size:0.82rem'>Predicts expected 10-over score based on powerplay performance</p>", unsafe_allow_html=True)

    if "Stage 3" in stage:
        st.markdown("<p class='stage-header'>STAGE 3 — AFTER 10 OVERS</p>", unsafe_allow_html=True)
        ca,cb = st.columns(2)
        with ca: pp_runs = st.number_input("PP Runs (0-6 ov)",    0, 120, 50, key="pr")
        with cb: pp_wkts = st.number_input("PP Wickets (0-6 ov)", 0,  10,  1, key="pw")
        cc,cd = st.columns(2)
        with cc: mid_runs = st.number_input("Total Runs (10 ov)",    0, 200, 95, key="mr")
        with cd: mid_wkts = st.number_input("Total Wickets (10 ov)", 0,  10,  3, key="mw")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p class='lbl'>Your Over/Under Target</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#888;font-size:0.82rem'>Type a score — model will say OVER or UNDER</p>", unsafe_allow_html=True)
        user_target = st.number_input("Your target score", 50, 300, 165, key="ut")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("PREDICT", key="btn_score"):
        if "Stage 1" in stage:
            pred, mae = stage_A(bat_team, bat_venue, bat_season)
            st.markdown(f"""
            <div class="score-box">
                <div class="lbl">PREDICTED 6-OVER POWERPLAY SCORE</div>
                <div class="score-num">{pred}</div>
                <div style="color:#888;font-size:0.82rem">
                    Expected range: {pred-mae} - {pred+mae} | MAE +/-{mae} runs
                </div>
            </div>
            <div>
                <span class="chip">{bat_team}</span>
                <span class="chip">{CITIES.get(bat_venue,"Neutral")}</span>
                <span class="chip">Season {bat_season}</span>
            </div>""", unsafe_allow_html=True)

        elif "Stage 2" in stage:
            pred, mae = stage_B(bat_team, bat_venue, bat_season, pp_runs, pp_wkts)
            crr = round(pp_runs/6, 2)
            st.markdown(f"""
            <div class="score-box">
                <div class="lbl">PREDICTED 10-OVER SCORE</div>
                <div class="score-num">{pred}</div>
                <div style="color:#888;font-size:0.82rem">
                    Expected range: {pred-mae} - {pred+mae} | MAE +/-{mae} runs
                </div>
            </div>
            <div>
                <span class="chip">{bat_team}</span>
                <span class="chip">PP: {pp_runs}/{pp_wkts}</span>
                <span class="chip">CRR: {crr}</span>
            </div>""", unsafe_allow_html=True)

        elif "Stage 3" in stage:
            pred, mae = stage_C(bat_team, bat_venue, bat_season,
                                pp_runs, pp_wkts, mid_runs, mid_wkts)
            show_over_under(pred, mae, user_target, "FINAL SCORE")
            crr = round(mid_runs/10, 2)
            rrr = round((user_target - mid_runs) / 10, 2)
            st.markdown(f"""
            <div style="margin-top:0.5rem">
                <span class="chip">{bat_team}</span>
                <span class="chip">PP: {pp_runs}/{pp_wkts}</span>
                <span class="chip">10ov: {mid_runs}/{mid_wkts}</span>
                <span class="chip">CRR: {crr}</span>
                <span class="chip">RRR to target: {rrr}</span>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#333;font-size:0.7rem;letter-spacing:2px'>XGBOOST · 1090 IPL MATCHES · 2008-2024</p>",
            unsafe_allow_html=True)
