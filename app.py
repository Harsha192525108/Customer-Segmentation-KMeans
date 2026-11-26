import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")
st.title("Customer Segmentation & Behavioural Pattern Analysis")
st.caption("K-Means clustering for data-driven marketing insights")

@st.cache_data
def make_sample(n=500, seed=17):
    rng=np.random.default_rng(seed)
    profiles=[(25,28,22,30,2,650),(42,65,72,78,8,2400),(55,38,48,35,4,1200),(32,82,95,88,11,3600)]
    groups=rng.choice(4,n,p=[.25,.30,.25,.20])
    rows=[]
    for i,g in enumerate(groups):
        age,inc,score,freq,orders,aov=profiles[g]
        rows.append({"CustomerID":f"C{i+1:04d}","Age":max(18,int(rng.normal(age,7))),
          "Annual Income (k$)":max(10,round(rng.normal(inc,10),1)),
          "Spending Score (1-100)":int(np.clip(rng.normal(score,10),1,100)),
          "Purchase Frequency":max(1,int(rng.normal(freq,2))),
          "Average Order Value":max(100,round(rng.normal(aov,aov*.18),2)),
          "Total Spend":max(100,round(rng.normal(orders*aov,orders*aov*.2),2)),
          "Recency (days)":max(1,int(rng.normal(35 if g in [1,3] else 75,18))),
          "Gender":rng.choice(["Female","Male"])})
    return pd.DataFrame(rows)

with st.sidebar:
    st.header("Project controls")
    upload=st.file_uploader("Upload customer CSV",type=["csv"])
    sample=st.checkbox("Use built-in sample dataset",value=upload is None)
    k=st.slider("Number of segments (K)",2,10,4)
    seed=st.number_input("Random seed",0,9999,42)

if upload is not None:
    try: df=pd.read_csv(upload)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}"); st.stop()
elif sample: df=make_sample()
else:
    st.info("Upload a CSV or select the sample dataset in the sidebar."); st.stop()
if df.empty:
    st.error("The dataset is empty."); st.stop()

st.header("Module 1: Customer Data Processing and Segmentation")
a,b,c=st.columns(3)
a.metric("Total customers",f"{len(df):,}")
b.metric("Data columns",len(df.columns))
c.metric("Missing values",int(df.isna().sum().sum()))
with st.expander("Preview customer records"): st.dataframe(df.head(20),use_container_width=True)

numeric=df.select_dtypes(include=np.number).columns.tolist()
if len(numeric)<2:
    st.error("At least two numeric columns are required."); st.stop()
defaults=[x for x in numeric if any(t in x.lower() for t in ["income","spending","frequency","order","spend","age","recency"])]
if len(defaults)<2: defaults=numeric[:min(4,len(numeric))]
features=st.multiselect("Select numeric features",numeric,default=defaults)
if len(features)<2:
    st.warning("Select at least two features."); st.stop()
data=df[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
mask=data.notna().all(axis=1)
clean=data.loc[mask]
if len(clean)<=k:
    st.error("Not enough complete rows for this K value. Reduce K or check your data."); st.stop()
scaled=StandardScaler().fit_transform(clean)
labels=KMeans(n_clusters=int(k),random_state=int(seed),n_init=10).fit_predict(scaled)
result=df.loc[mask].copy()
result["Customer Segment"]=labels+1
score=silhouette_score(scaled,labels) if len(set(labels))>1 else float("nan")
x,y,z=st.columns(3)
x.metric("Segments created",k)
y.metric("Customers analysed",len(result))
z.metric("Silhouette score","N/A" if np.isnan(score) else f"{score:.3f}")
counts=result["Customer Segment"].value_counts().sort_index().rename_axis("Segment").reset_index(name="Customers")
st.plotly_chart(px.bar(counts,x="Segment",y="Customers",color="Segment",text="Customers",title="Customer count by segment"),use_container_width=True)
coords=PCA(n_components=2).fit_transform(scaled)
plot=pd.DataFrame({"Component 1":coords[:,0],"Component 2":coords[:,1],"Segment":labels+1})
st.plotly_chart(px.scatter(plot,x="Component 1",y="Component 2",color="Segment",title="Customer segments (PCA visualisation)"),use_container_width=True)

st.header("Module 2: Behavioural Pattern Analysis and Marketing Insights")
summary=result.groupby("Customer Segment")[features].mean().round(2)
summary.insert(0,"Customer Count",result.groupby("Customer Segment").size())
st.write("Average customer characteristics for each segment")
st.dataframe(summary,use_container_width=True)
col1,col2=st.columns(2)
with col1: xfeature=st.selectbox("Behaviour feature (X-axis)",features,index=0)
with col2: yfeature=st.selectbox("Behaviour feature (Y-axis)",features,index=1)
st.plotly_chart(px.scatter(result,x=xfeature,y=yfeature,color="Customer Segment",hover_data=[c for c in ["CustomerID","Gender"] if c in result.columns],title="Behavioural relationship across segments"),use_container_width=True)

def recommendation(row):
    income=next((c for c in features if "income" in c.lower()),None)
    spend=next((c for c in features if "spend" in c.lower()),None)
    freq=next((c for c in features if "frequency" in c.lower()),None)
    recency=next((c for c in features if "recency" in c.lower()),None)
    if income and spend:
        if row[income]>=result[income].median() and row[spend]>=result[spend].median():
            return "High-value customers","Premium offers, loyalty rewards and early product access"
        if row[income]>=result[income].median():
            return "High-income, low-spending","Personalised product suggestions and introductory offers"
        if row[spend]>=result[spend].median():
            return "Value-conscious active spenders","Value bundles, discounts and repeat-purchase rewards"
    if freq and row[freq]>=result[freq].median():
        return "Frequent customers","Subscriptions, loyalty points and referral benefits"
    if recency and row[recency]>result[recency].median():
        return "At-risk / inactive customers","Win-back campaigns and relevant reminders"
    return "General / developing segment","Test targeted campaigns and monitor customer response"

insights=[]
for seg,row in result.groupby("Customer Segment")[features].mean().iterrows():
    profile,action=recommendation(row)
    insights.append({"Segment":int(seg),"Customer profile":profile,"Suggested marketing action":action})
st.subheader("Segment-wise marketing recommendations")
st.dataframe(pd.DataFrame(insights),use_container_width=True,hide_index=True)
st.download_button("Download segmented customer data",result.to_csv(index=False).encode("utf-8"),"customer_segments.csv","text/csv")
st.download_button("Download segment summary",summary.to_csv().encode("utf-8"),"segment_summary.csv","text/csv")
st.caption("Academic demonstration using K-Means. The built-in sample data is synthetic; validate segment meaning before applying campaigns.")
