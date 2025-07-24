import pandas as pd
from pathlib import Path

def merge_station_data(
        meter_path: str,
        instance_path: str,
        station_id: int,
        out_dir: str = '.',
        chunksize: int | None = None
    ) -> Path:
    """
    合并两个 CSV，生成指定场站的清洗结果。

    Parameters
    ----------
    meter_path : str         # vpp_meter.csv 路径
    instance_path : str      # vpp_instance_power.csv 路径
    station_id : int         # 场站ID
    out_dir : str            # 结果保存目录
    chunksize : int | None   # pandas 读取块大小，None 表示一次性读入

    Returns
    -------
    Path : 输出文件路径
    """
    # -------- 1. 读取 vpp_meter.csv --------
    meter_cols = ['ts', 'station_ref_id', 'total_active_power']
    # auto parse ts to datetime for safety
    meter_df = pd.read_csv(
        meter_path,
        usecols=meter_cols,
        parse_dates=['ts'],
        chunksize=chunksize
    )

    # 如果指定 chunksize，则需拼接块；否则 meter_df 本身是 DataFrame
    if chunksize:
        meter_df = pd.concat(
            chunk[chunk['station_ref_id'] == station_id] for chunk in meter_df
        )
    else:
        meter_df = meter_df[meter_df['station_ref_id'] == station_id]

    # -------- 2. 读取 vpp_instance_power.csv 并聚合 --------
    inst_cols = ['ts', 'station_ref_id', 'power_send', 'power_use']
    inst_iter = pd.read_csv(
        instance_path,
        usecols=inst_cols,
        parse_dates=['ts'],
        chunksize=chunksize
    )

    inst_list = []
    for chunk in inst_iter:
        filtered = chunk[chunk['station_ref_id'] == station_id]
        # 按 ts 聚合
        g = (
            filtered
            .groupby(['ts', 'station_ref_id'], as_index=False)
            .agg(send_sum=('power_send', 'sum'),
                 use_sum=('power_use',  'sum'))
        )
        inst_list.append(g)

    instance_df = pd.concat(inst_list, ignore_index=True) if inst_list else pd.DataFrame(
        columns=['ts', 'station_ref_id', 'send_sum', 'use_sum'])

    # -------- 3. 合并 & 计算 not_use_power --------
    merged = pd.merge(
        meter_df,
        instance_df,
        on=['ts', 'station_ref_id'],
        how='inner'   # 仅取两表同时存在的数据
    )

    merged['not_use_power'] = (
        merged['total_active_power'] + merged['send_sum'] - merged['use_sum']
    )

    # 仅保留需求字段
    result = merged[['ts', 'station_ref_id', 'total_active_power', 'not_use_power']]

    # -------- 4. 输出 --------
    out_path = Path(out_dir).joinpath(f'merged_station_{station_id}.csv')
    result.to_csv(out_path, index=False)
    print(f'Done! rows={len(result)} -> {out_path}')
    return out_path


if __name__ == '__main__':
    # === 参数区 ===
    STATION_ID = 1716387625733984256
    METER_FILE = 'vpp_meter.csv'
    INSTANCE_FILE = 'vpp_instance_power.csv'
    OUTPUT_DIR = ''
    CHUNKSIZE = 500_000      # 大文件可调，内存够则设 None

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    merge_station_data(
        meter_path=METER_FILE,
        instance_path=INSTANCE_FILE,
        station_id=STATION_ID,
        out_dir=OUTPUT_DIR,
        chunksize=CHUNKSIZE
    )