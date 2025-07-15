# main_api.py
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import holidays
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# --- 0. 初始化 FastAPI 应用 ---
app = FastAPI(
    title="负荷预测 API (Load Forecasting API)",
    description="一个使用LSTM模型预测未来96个点（24小时）负荷的API。",
    version="1.1.0"
)

warnings.filterwarnings("ignore")

# --- 1. 加载模型和预处理器 ---
MODEL_PATH = 'output/model_global.h5'
SCALER_X_PATH = 'output/scaler_x_global.pkl'
SCALER_Y_PATH = 'output/scaler_y_global.pkl'
LABEL_ENCODER_PATH = 'output/label_encoder.pkl'

try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'mae': tf.keras.losses.MeanAbsoluteError}
    )
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("模型和预处理器加载成功。")
except Exception as e:
    print(f"致命错误：加载模型或预处理器失败，API无法启动。错误: {e}")
    model, scaler_x, scaler_y, le = None, None, None, None

# --- 2. 定义特征工程函数 ---
cn_holidays = holidays.country_holidays('CN')

def create_features_for_prediction(df):
    """
    为预测数据创建所有必需的特征。
    注意：is_work 和 is_peak 由调用者提供，这里不再计算。
    """
    df = df.copy()
    df['energy_date'] = pd.to_datetime(df['energy_date'])
    df['hour'] = df['energy_date'].dt.hour
    df['weekday'] = df['energy_date'].dt.weekday
    df['day'] = df['energy_date'].dt.day
    df['month'] = df['energy_date'].dt.month
    for lag in [1, 4, 96, 192]:
        df[f'load_lag{lag}'] = df['load_discharge_delta'].shift(lag)
    for win in [4, 12, 96, 192]:
        df[f'load_ma{win}'] = df['load_discharge_delta'].rolling(win, min_periods=1).mean()
        df[f'load_std{win}'] = df['load_discharge_delta'].rolling(win, min_periods=1).std()
    df['is_holiday'] = df['energy_date'].isin(cn_holidays).astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_wday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_wday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    # ** 已移除 **: is_work 和 is_peak 的自动计算逻辑，因为它们现在由用户传入。
    return df

# --- 3. 定义API的数据模型 (Pydantic) ---

class HistoricalDataPoint(BaseModel):
    energy_date: str = Field(..., example="2025-07-10T00:00:00Z", description="ISO 8601 格式的时间戳")
    load_discharge_delta: float = Field(..., example=150.5, description="历史负荷值")
    temp: float = Field(..., example=25.2, description="历史温度")
    code: int = Field(..., example=100, description="历史天气代码")
    humidity: float = Field(..., example=60.1, description="历史湿度")
    windSpeed: float = Field(..., example=10.3, description="历史风速")
    cloud: float = Field(..., example=25.0, description="历史云量")
    is_work: int = Field(..., example=1, description="是否为工作日 (1: 是, 0: 否)") # 新增字段
    is_peak: int = Field(..., example=0, description="是否为高峰时段 (1: 是, 0: 否)") # 新增字段

class FutureDataPoint(BaseModel):
    energy_date: str = Field(..., example="2025-07-15T00:00:00Z", description="ISO 8601 格式的时间戳")
    temp: float = Field(..., example=22.5, description="预测日期的温度")
    code: int = Field(..., example=101, description="预测日期的天气代码")
    humidity: float = Field(..., example=66.0, description="预测日期的湿度")
    windSpeed: float = Field(..., example=7.9, description="预测日期的风速")
    cloud: float = Field(..., example=32.0, description="预测日期的云量")
    is_work: int = Field(..., example=1, description="是否为工作日 (1: 是, 0: 否)") # 新增字段
    is_peak: int = Field(..., example=0, description="是否为高峰时段 (1: 是, 0: 否)") # 新增字段

class PredictionRequest(BaseModel):
    station_ref_id: str = Field(..., example="STATION_001", description="需要预测的站点ID")
    historical_data: List[HistoricalDataPoint] = Field(..., description="预测日期前5天（480个点）的历史数据")
    future_data: List[FutureDataPoint] = Field(..., description="预测当天（96个点）的天气预报及工况数据")

class PredictionPoint(BaseModel):
    timestamp: str
    predicted_load: float

class PredictionResponse(BaseModel):
    prediction_date: str
    predictions: List[PredictionPoint]

# --- 4. 定义API端点 ---
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_load(request: PredictionRequest):
    # ... (内部逻辑与之前版本基本一致，但现在会使用传入的is_work/is_peak) ...
    # ... 为简洁起见，此处省略与之前版本重复的代码 ...
    if not all([model, scaler_x, scaler_y, le]):
        raise HTTPException(status_code=500, detail="模型或预处理器未加载，服务不可用。")

    past_steps_required = model.input_shape[1]
    future_steps_required = 96
    if len(request.historical_data) != past_steps_required:
        raise HTTPException(status_code=422, detail=f"数据错误：'historical_data' 需要提供 {past_steps_required} 个点，但收到了 {len(request.historical_data)} 个。")
    if len(request.future_data) != future_steps_required:
        raise HTTPException(status_code=422, detail=f"数据错误：'future_data' 需要提供 {future_steps_required} 个点，但收到了 {len(request.future_data)} 个。")

    try:
        station_id = request.station_ref_id
        hist_df = pd.DataFrame([p.model_dump() for p in request.historical_data])
        future_df = pd.DataFrame([p.model_dump() for p in request.future_data])
        
        future_df['load_discharge_delta'] = np.nan
        full_df = pd.concat([hist_df, future_df], ignore_index=True)
        full_df['station_ref_id'] = station_id
        
        df_featured = create_features_for_prediction(full_df)
        df_featured = df_featured.fillna(method='ffill').fillna(method='bfill')

        if df_featured.isnull().values.any():
            raise HTTPException(status_code=400, detail="数据处理后仍包含NaN值，请检查输入数据的完整性。")

        df_featured['station_enc'] = le.transform([station_id])[0]
        
        feature_cols = scaler_x.feature_names_in_
        df_final = df_featured[feature_cols]
        X_scaled = scaler_x.transform(df_final)
        
        input_sequence = X_scaled[-(past_steps_required + future_steps_required):-future_steps_required]
        input_tensor = np.expand_dims(input_sequence, axis=0)

        pred_scaled = model.predict(input_tensor, verbose=0)[0]
        pred_unscaled = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        prediction_timestamps = pd.to_datetime(future_df['energy_date'])
        prediction_date = prediction_timestamps.iloc[0].strftime('%Y-%m-%d')
        
        response_predictions = []
        for i, timestamp in enumerate(prediction_timestamps):
            response_predictions.append({
                "timestamp": timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "predicted_load": float(round(pred_unscaled[i], 4))
            })
            
        return PredictionResponse(prediction_date=prediction_date, predictions=response_predictions)
    except ValueError as ve:
        if "was not seen" in str(ve) or "y contains new labels" in str(ve):
             raise HTTPException(status_code=400, detail=f"站点ID '{station_id}' 无效，因为它不在训练数据中。")
        raise HTTPException(status_code=500, detail=f"内部处理错误: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发生未知服务器错误: {e}")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "欢迎使用负荷预测API，请访问 /docs 查看API文档。"}