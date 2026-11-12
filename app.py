import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

st.set_page_config(page_title="FollicleCare | Hair & Scalp Health", page_icon="🌿", layout="wide")
st.markdown("""
<style>
.stApp{background:#f7fbf7;color:#20352a}.block-container{max-width:1400px;padding-top:1.3rem}
h1,h2,h3,h4,p,label,li{color:#20352a}
.hero{background:linear-gradient(115deg,#fff,#eaf6ec);border:1px solid #d6e7d9;border-left:6px solid #75bd86;border-radius:18px;padding:25px 30px;margin-bottom:18px}
.hero h1{color:#245b3c;margin:0;font-size:2.3rem}.hero p{color:#405f4b;margin:.45rem 0 0}
.eyebrow{color:#39734d;font-size:.76rem;letter-spacing:1.5px;font-weight:800}
.panel,.kpi{background:#fff;border:1px solid #d6e7d9;border-radius:15px;padding:17px}
.kpi-label{color:#536b5b;font-size:.82rem;font-weight:650}.kpi-value{color:#245b3c;font-size:1.65rem;font-weight:800}
.kpi-note{color:#536b5b;font-size:.76rem}
.stTabs [data-baseweb="tab"]{background:#eaf4eb;color:#245b3c;border-radius:9px 9px 0 0;padding:10px 13px}
.stTabs [aria-selected="true"]{background:#d7eedb!important;color:#194b30!important}
[data-testid="stSidebar"]{background:#eff7f0;border-right:1px solid #d6e7d9}
[data-testid="stMetric"]{background:#fff;border:1px solid #d6e7d9;padding:12px;border-radius:12px}
.stButton button,.stDownloadButton button{background:#3f8759;color:white;border:1px solid #3f8759;border-radius:9px}
.stButton button:hover,.stDownloadButton button:hover{background:#245b3c;color:white}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="eyebrow">HAIR & SCALP HEALTH · PROBABILITY INSIGHTS</div>
<h1>🌿 FollicleCare</h1><p>A thoughtful space to explore hair and scalp health through structured observations, image review and calibrated machine-learning probabilities.</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## FollicleCare")
    st.caption("Hair & scalp health insights")
    st.markdown("---")
    st.write("Structured symptom profile")
    st.write("Optional scalp image upload")
    st.caption("Educational information only. Not a medical diagnosis.")

@st.cache_resource
def train():
    X,y=make_classification(n_samples=2400,n_features=12,n_informative=8,n_redundant=2,
        weights=[.62,.38],class_sep=.9,flip_y=.04,random_state=27)
    names=["Shedding frequency","Visible thinning","Scalp itching","Flaking","Redness","Tenderness",
           "Patchy hair loss","Recent stress","Recent illness","Family history","Hair breakage","Duration"]
    X=pd.DataFrame(X,columns=names); y=pd.Series(y)
    xa,xb,ya,yb=train_test_split(X,y,test_size=.25,stratify=y,random_state=27)
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500))
    rf=RandomForestClassifier(n_estimators=250,min_samples_leaf=3,random_state=27)
    lr.fit(xa,ya); rf.fit(xa,ya)
    lc=CalibratedClassifierCV(lr,method="sigmoid",cv=5)
    rc=CalibratedClassifierCV(rf,method="isotonic",cv=5)
    lc.fit(xa,ya); rc.fit(xa,ya)
    return xb,yb,{"Logistic Regression · raw":lr,"Random Forest · raw":rf,
                  "Logistic Regression · Platt":lc,"Random Forest · isotonic":rc},names

def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); out=0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any(): out += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)

Xtest,ytest,models,features=train()
probs={n:m.predict_proba(Xtest)[:,1] for n,m in models.items()}
rows=[]
for n,p in probs.items():
    rows.append({"Model":n,"ROC-AUC":roc_auc_score(ytest,p),"Brier score":brier_score_loss(ytest,p),
                 "ECE":ece(ytest,p),"Accuracy":accuracy_score(ytest,(p>=.5).astype(int))})
metrics=pd.DataFrame(rows)

tabs=st.tabs(["Overview","Hair health check-in","Image review","Model performance","Calibration lab","About"])
with tabs[0]:
    st.subheader("A clearer view of hair and scalp concerns")
    st.write("Hair changes can have many causes. FollicleCare organizes common observations and demonstrates how probability models communicate uncertainty.")
    cs=st.columns(4)
    for c,l,v,n in zip(cs,["Input indicators","Model families","Calibration methods","Evaluation measures"],["12","2","2","4"],["structured features","linear + tree-based","Platt + isotonic","AUC, Brier, ECE, accuracy"]):
        c.markdown(f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div><div class="kpi-note">{n}</div></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    a.markdown('<div class="panel"><b>Hair health check-in</b><p>Record visible changes and symptoms using a guided form.</p></div>',unsafe_allow_html=True)
    b.markdown('<div class="panel"><b>Image review</b><p>Preview a scalp photograph and record observations.</p></div>',unsafe_allow_html=True)
    st.info("The model experiment uses simulated data to demonstrate calibration. Scores are not clinical risk estimates.")
with tabs[1]:
    st.subheader("Hair & scalp check-in")
    st.write("Select observations that best describe the current concern. This is an educational interaction, not a diagnosis.")
    with st.form("checkin"):
        c1,c2=st.columns(2)
        opts=["None","Mild","Moderate","High"]
        with c1:
            shedding=st.select_slider("Hair shedding",options=["Not noticeable","Mild","Moderate","High","Very high"],value="Moderate")
            thinning=st.select_slider("Visible thinning",options=["None","Slight","Moderate","Marked"],value="Slight")
            itching=st.select_slider("Scalp itching",options=opts,value="None")
            flakes=st.select_slider("Flaking",options=opts,value="None")
            redness=st.select_slider("Redness",options=opts,value="None")
            tenderness=st.select_slider("Tenderness",options=opts,value="None")
        with c2:
            patchy=st.selectbox("Patchy hair loss",["No","Unsure","Yes"])
            stress=st.selectbox("Significant recent stress",["No","Unsure","Yes"])
            illness=st.selectbox("Recent illness / health change",["No","Unsure","Yes"])
            family=st.selectbox("Family history of thinning",["No","Unsure","Yes"])
            breakage=st.select_slider("Hair breakage",options=opts,value="Mild")
            duration=st.selectbox("Duration",["Less than 1 month","1–3 months","3–6 months","More than 6 months"])
        submitted=st.form_submit_button("Generate illustrative score")
    if submitted:
        mp={"Not noticeable":0,"None":0,"No":0,"Unsure":1,"Slight":1,"Mild":1,"Yes":2,
            "Moderate":2,"High":3,"Very high":4,"Marked":3,
            "Less than 1 month":0,"1–3 months":1,"3–6 months":2,"More than 6 months":3}
        vals=[mp[shedding],mp[thinning],mp[itching],mp[flakes],mp[redness],mp[tenderness],
              mp[patchy],mp[stress],mp[illness],mp[family],mp[breakage],mp[duration]]
        profile=pd.DataFrame([vals],columns=features)
        score=float(models["Logistic Regression · Platt"].predict_proba(profile)[0,1])
        st.metric("Illustrative model score",f"{score:.1%}")
        st.caption("This is generated from simulated training data. It is not the probability that you have a hair or scalp disease.")
        if patchy=="Yes" or redness=="High" or tenderness=="High":
            st.warning("Persistent patchy loss, marked redness or pain deserves assessment by a qualified dermatologist.")
        else:
            st.info("If hair changes persist, worsen or concern you, consider a professional assessment.")
with tabs[2]:
    st.subheader("Scalp image review")
    st.write("Upload a photograph to preview it and record observations for discussion with a healthcare professional.")
    img=st.file_uploader("Upload scalp or hair photograph",type=["jpg","jpeg","png"])
    if img:
        a,b=st.columns(2)
        a.image(img,caption="Uploaded image",use_container_width=True)
        b.markdown('<div class="panel"><b>Image handling</b><p>This version previews the image locally in the app. It does not run a validated dermatology image classifier or send the image to an external service.</p></div>',unsafe_allow_html=True)
        st.text_area("Observation notes",placeholder="Visible flaking, parting width, redness, area of concern…")
    else:
        st.info("Image upload is currently a preview and note-taking feature. A disease classifier requires a suitable licensed, expert-labelled dataset.")
with tabs[3]:
    st.subheader("Model performance")
    st.write("All model variants are evaluated on the same held-out test partition.")
    st.dataframe(metrics.style.format({"ROC-AUC":"{:.3f}","Brier score":"{:.3f}","ECE":"{:.3f}","Accuracy":"{:.3f}"}),use_container_width=True,hide_index=True)
    metric=st.selectbox("Metric",["ROC-AUC","Brier score","ECE","Accuracy"])
    fig=px.bar(metrics,x="Model",y=metric,color="Model",text_auto=".3f",title=f"{metric} across model versions")
    fig.update_layout(showlegend=False,xaxis_tickangle=-15,height=400)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Results use simulated data and do not indicate clinical performance.")
with tabs[4]:
    st.subheader("Probability calibration")
    st.write("A reliability diagram compares average predicted probability with observed event frequency. Ideal calibration follows the diagonal.")
    chosen=st.multiselect("Models",list(models),default=list(models))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Ideal calibration",line=dict(dash="dash",color="#92aa96")))
    for n in chosen:
        frac,mean=calibration_curve(ytest,probs[n],n_bins=10,strategy="uniform")
        fig.add_trace(go.Scatter(x=mean,y=frac,mode="lines+markers",name=n))
    fig.update_layout(title="Reliability diagram",xaxis_title="Mean predicted probability",yaxis_title="Observed event frequency",
        xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]),height=460,legend=dict(orientation="h",y=-.25))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Expected Calibration Error (ECE)**")
    st.latex(r"ECE = \sum_{b=1}^{B}\frac{|S_b|}{N}|acc(S_b)-conf(S_b)|")
    st.write("ECE summarizes the weighted gap between confidence and observed frequency across bins. Its value depends on binning and should be read alongside the reliability diagram and Brier score.")
    st.dataframe(metrics[["Model","ECE","Brier score"]].sort_values("ECE"),use_container_width=True,hide_index=True)
with tabs[5]:
    st.subheader("Methodology & responsible use")
    st.markdown("### Model experiment")
    st.write("Logistic Regression and Random Forest are trained on a reproducible simulated binary dataset. Sigmoid calibration is applied to Logistic Regression and isotonic calibration to Random Forest using cross-validation on training data. A separate test partition is used for evaluation.")
    st.markdown("### Why calibration matters")
    st.write("A model can rank examples correctly while overstating or understating its confidence. Calibration checks whether predicted probabilities align with observed frequencies.")
    st.markdown("### Data transparency")
    st.warning("The current model data are simulated. They do not represent patients, scalp photographs or disease prevalence. This application must not be described as a validated hair-loss predictor or disease detector.")
    st.markdown("### Future research")
    st.write("A genuine clinical version requires an appropriately licensed and representative dataset, expert-verified labels, participant-aware data splits, independent validation and dermatology oversight.")
    st.markdown("### When to seek care")
    st.write("Sudden or patchy hair loss, scalp pain, swelling, sores, severe itching or rapidly worsening symptoms should be assessed by a qualified healthcare professional.")
    st.download_button("Download model metrics CSV",metrics.to_csv(index=False).encode(),file_name="folliclecare_model_metrics.csv",mime="text/csv")
st.markdown("---")
st.markdown("<div style='text-align:center;color:#536b5b;font-size:.82rem;'>FollicleCare · Hair & Scalp Health Insights</div>",unsafe_allow_html=True)
