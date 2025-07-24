# =========================================================
#  Example script to train a single station model
#  This demonstrates the complete workflow
# =========================================================
import os
import pandas as pd
from train_encdec_7d_single_station import main as train_main

def get_first_station_id(data_file='merged_station_test.csv'):
    """Get the first station ID from the data file"""
    try:
        # Read just the first few rows to get station ID
        df = pd.read_csv(data_file, nrows=100)
        if 'station_ref_id' in df.columns:
            first_station = df['station_ref_id'].iloc[0]
            print(f"📊 First station ID found: {first_station}")
            return str(first_station)
        else:
            print("❌ No 'station_ref_id' column found in data")
            return None
    except Exception as e:
        print(f"❌ Error reading data file: {str(e)}")
        return None

def main():
    """Main function to demonstrate training workflow"""
    print("🚀 Single Station Model Training Example")
    print("=" * 50)
    
    # Check if data file exists
    data_file = 'merged_station_test.csv'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("   Please make sure the data file exists in the current directory")
        return
    
    # Get first station ID
    station_id = get_first_station_id(data_file)
    if not station_id:
        print("❌ Could not determine station ID from data file")
        return
    
    print(f"🎯 Training model for station: {station_id}")
    print(f"🎯 Training model for station: {station_id}")
    print("⏳ Starting training process...")
    
    try:
        # Train the model
        output_dir = train_main(station_id, data_file)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        print("\n📋 Next steps:")
        print("1. Start the API server: python api_encdec_7d_single_station.py")
        print("2. Test the API: python test_api_encdec_7d_single_station.py")
        print("3. For incremental fine-tuning, use: python incremental_finetune.py")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("- Check if the data file has sufficient data points")
        print("- Ensure all required columns are present")
        print("- Check for data quality issues (missing values, etc.)")
        return None

if __name__ == "__main__":
    main()
