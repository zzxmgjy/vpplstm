"""
api_encdec_7d.py  ——  FastAPI + is_work / is_peak
"""
import os, joblib, holidays, uvicorn, torch, torch.nn as nn
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

ROOT='output_pytorch'
PAST_STEPS=96*5; FUTURE_STEPS=96*7

ENC_COLS=[ 'temp','temp_squared','humidity','windSpeed',
           'load_lag1','load_lag4','load_lag24','load_lag96',
           'load_ma4','load_ma24','load_ma96',
           'load_std4','load_std24','load_std96',
           'is_holiday','is_work','is_peak',
           'sin_hour','cos_hour','sin_wday','cos_wday']
DEC_COLS=[ 'temp','humidity','windSpeed',
           'is_holiday','is_work','is_peak',
           'sin_hour','cos_hour','sin_wday','cos_wday']

sc_enc=joblib.load(f'{ROOT}/scaler_enc.pkl')
sc_dec=joblib.load(f'{ROOT}/scaler_dec.pkl')
sc_y  =joblib.load(f'{ROOT}/scaler_y.pkl')

class EncDec(nn.Module):
    def __init__(self,d1,d2,hid,fut,drop):
        super().__init__()
        self.enc=nn.LSTM(d1,hid,batch_first=True)
        self.dec=nn.LSTM(d2,hid,batch_first=True)
        self.dp=nn.Dropout(drop)
        self.fc=nn.Linear(hid,1)
    def forward(self,xe,xd):
        _,(h,c)=self.enc(xe)
        y,_=self.dec(xd,(h,c))
        return self.fc(self.dp(y)).squeeze(-1)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model=EncDec(len(ENC_COLS),len(DEC_COLS),128,FUTURE_STEPS,.24).to(device)
base_model.load_state_dict(torch.load(f'{ROOT}/model_base_encdec7d.pth',map_location=device))
base_model.eval()
model_cache={}

# ---------- Pydantic ----------
class Past(BaseModel):
    energy_date: datetime
    load_discharge_delta: float
    temp: float; humidity: float; windSpeed: float
    is_work: int | None = None
    is_peak: int | None = None
class Fut(BaseModel):
    energy_date: datetime
    temp: float; humidity: float; windSpeed: float
    is_work: int | None = None
    is_peak: int | None = None
class Req(BaseModel):
    station_id: str
    past_data: List[Past]  = Field(..., description="480 条")
    future_external: List[Fut]= Field(..., description="672 条")
class Item(BaseModel):
    energy_date: datetime; load_discharge_delta_pred: float
class Resp(BaseModel):
    station_id: str; model_used: str; predictions: List[Item]

# ---------- Utils ----------
cn_holidays=holidays.country_holidays('CN')
def enrich(df):
    df['hour']=df['energy_date'].dt.hour
    df['weekday']=df['energy_date'].dt.weekday
    df['temp_squared']=df['temp']**2
    df['sin_hour']=np.sin(2*np.pi*df['hour']/24)
    df['cos_hour']=np.cos(2*np.pi*df['hour']/24)
    df['sin_wday']=np.sin(2*np.pi*df['weekday']/7)
    df['cos_wday']=np.cos(2*np.pi*df['weekday']/7)
    df['is_holiday']=df['energy_date'].isin(cn_holidays).astype(int)
    if 'is_work' not in df.columns:
        df['is_work']=((df['weekday']<5)&(df['is_holiday']==0)).astype(int)
    if 'is_peak' not in df.columns:
        df['is_peak']=(df['hour'].between(8,18)|df['hour'].between(18,21)).astype(int)
    return df
def build_encoder(p):
    p=enrich(p.sort_values('energy_date'))
    p['load_lag1']=p['load_discharge_delta'].shift(1)
    p['load_lag4']=p['load_discharge_delta'].shift(4)
    p['load_lag24']=p['load_discharge_delta'].shift(24)
    p['load_lag96']=p['load_discharge_delta'].shift(96)
    for w in [4,24,96]:
        p[f'load_ma{w}']=p['load_discharge_delta'].rolling(w,1).mean()
        p[f'load_std{w}']=p['load_discharge_delta'].rolling(w,1).std()
    p=p.fillna(method='ffill').dropna()
    if len(p)<PAST_STEPS: raise ValueError('历史不足 480')
    enc=sc_enc.transform(p.tail(PAST_STEPS)[ENC_COLS])
    return torch.from_numpy(enc.astype(np.float32)).unsqueeze(0)
def build_decoder(f):
    f=enrich(f.sort_values('energy_date'))
    if len(f)!=FUTURE_STEPS: raise ValueError('future_external 需要 672 条')
    dec=sc_dec.transform(f[DEC_COLS])
    return torch.from_numpy(dec.astype(np.float32)).unsqueeze(0)
def load_model(sid):
    if sid in model_cache: return model_cache[sid],True
    p=os.path.join(ROOT,f'mode_{sid}',f'model_optimized_{sid}.pth')
    if os.path.exists(p):
        m=EncDec(len(ENC_COLS),len(DEC_COLS),128,FUTURE_STEPS,.24).to(device)
        m.load_state_dict(torch.load(p,map_location=device)); m.eval()
        model_cache[sid]=m; return m,True
    return base_model,False

app=FastAPI(title='7-Day API (is_work & is_peak)')

@app.post('/predict',response_model=Resp)
def predict(req:Req):
    past_df=pd.DataFrame([x.dict() for x in req.past_data])
    fut_df =pd.DataFrame([x.dict() for x in req.future_external])
    try:
        xe=build_encoder(past_df).to(device)
        xd=build_decoder(fut_df).to(device)
    except ValueError as e:
        raise HTTPException(400,str(e))
    model,flag=load_model(req.station_id)
    with torch.no_grad():
        y_scaled=model(xe,xd).cpu().numpy().flatten()
    y=sc_y.inverse_transform(y_scaled.reshape(-1,1)).flatten()
    preds=[Item(energy_date=fut_df['energy_date'].iloc[i],
                load_discharge_delta_pred=float(v)) for i,v in enumerate(y)]
    return Resp(station_id=req.station_id,
                model_used='station' if flag else 'base',
                predictions=preds)

if __name__=='__main__':
    uvicorn.run('api_encdec_7d:app',host='0.0.0.0',port=8000)