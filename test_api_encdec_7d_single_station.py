# =========================================================
#  Test Script for Single Station Power Prediction API
#  Tests the API endpoints and validates responses
# =========================================================
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# API Configuration
API_BASE_URL = "http://localhost:5000"
TEST_STATION_ID = "1716387625733984256"  # Use the station ID from your CSV

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Loaded stations: {data['loaded_stations']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_list_stations():
    """Test the list stations endpoint"""
    print("\n🔍 Testing list stations endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/stations")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ List stations passed")
            print(f"   Available stations: {len(data['stations'])}")
            for station in data['stations']:
                print(f"   - Station {station['station_id']}: loaded={station['loaded']}")
            return data['stations']
        else:
            print(f"❌ List stations failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ List stations error: {str(e)}")
        return []

def test_load_model(station_id):
    """Test loading a model for a specific station"""
    print(f"\n🔍 Testing load model for station {station_id}...")
    try:
        response = requests.post(f"{API_BASE_URL}/load_model/{station_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Load model passed: {data['message']}")
            if 'station_info' in data:
                info = data['station_info']
                print(f"   Data points: {info.get('data_points', 'N/A')}")
                print(f"   Training samples: {info.get('training_samples', 'N/A')}")
                print(f"   Features count: {info.get('features_count', 'N/A')}")
            return True
        else:
            print(f"❌ Load model failed: {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Load model error: {str(e)}")
        return False

def generate_test_data(station_id, days=7, interval_minutes=15):
    """Generate test data for prediction"""
    print(f"\n📊 Generating test data for {days} days...")
    
    # Generate timestamps (7 days of historical data)
    end_time = datetime.now().replace(second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)
    
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    # Generate realistic power data with daily patterns
    data = []
    for i, ts in enumerate(timestamps):
        # Create daily pattern: higher during day, lower at night
        hour = ts.hour
        minute = ts.minute
        
        # Base load with daily pattern
        base_power = 300 + 200 * np.sin(2 * np.pi * (hour + minute/60) / 24 + np.pi/2)
        
        # Add weekly pattern (lower on weekends)
        weekday_factor = 0.8 if ts.weekday() >= 5 else 1.0
        
        # Add some random variation
        noise = np.random.normal(0, 20)
        
        total_power = max(0, base_power * weekday_factor + noise)
        not_use_power = total_power * (0.6 + 0.2 * np.random.random())  # 60-80% non-controllable
        
        data.append({
            'ts': ts.isoformat(),
            'total_active_power': round(total_power, 2),
            'not_use_power': round(not_use_power, 2),
            'temp': round(20 + 10 * np.sin(2 * np.pi * (hour + minute/60) / 24), 1),
            'humidity': round(50 + 20 * np.random.random(), 1),
            'windSpeed': round(5 + 5 * np.random.random(), 1)
        })
    
    print(f"✅ Generated {len(data)} data points")
    print(f"   Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"   Power range: {min(d['total_active_power'] for d in data):.1f} - {max(d['total_active_power'] for d in data):.1f}")
    
    return data

def test_prediction(station_id, test_data):
    """Test the prediction endpoint"""
    print(f"\n🔍 Testing prediction for station {station_id}...")
    
    # Prepare request payload
    payload = {
        'data': test_data
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/{station_id}",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        prediction_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Prediction time: {prediction_time:.2f} seconds")
            print(f"   Station ID: {data['station_id']}")
            print(f"   Prediction period: {data['prediction_start']} to {data['prediction_end']}")
            print(f"   Total prediction points: {data['total_points']}")
            
            # Analyze predictions
            predictions = data['predictions']
            power_values = [p['total_active_power'] for p in predictions]
            not_use_power_values = [p['not_use_power'] for p in predictions]
            
            print(f"   Power prediction range: {min(power_values):.1f} - {max(power_values):.1f}")
            print(f"   Not-use power range: {min(not_use_power_values):.1f} - {max(not_use_power_values):.1f}")
            
            # Show daily averages
            daily_averages = {}
            for pred in predictions:
                day = pred['day']
                if day not in daily_averages:
                    daily_averages[day] = {'power': [], 'not_use_power': []}
                daily_averages[day]['power'].append(pred['total_active_power'])
                daily_averages[day]['not_use_power'].append(pred['not_use_power'])
            
            print("   Daily averages:")
            for day in sorted(daily_averages.keys()):
                avg_power = np.mean(daily_averages[day]['power'])
                avg_not_use = np.mean(daily_averages[day]['not_use_power'])
                print(f"     Day {day}: Power={avg_power:.1f}, Not-use={avg_not_use:.1f}")
            
            # Metadata
            metadata = data.get('metadata', {})
            print(f"   Model type: {metadata.get('model_type', 'N/A')}")
            print(f"   Input data points: {metadata.get('input_data_points', 'N/A')}")
            
            return data
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            else:
                print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None

def save_prediction_results(prediction_data, filename="prediction_results.json"):
    """Save prediction results to file"""
    if prediction_data:
        print(f"\n💾 Saving prediction results to {filename}...")
        try:
            with open(filename, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            print(f"✅ Results saved successfully")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")

def create_prediction_csv(prediction_data, filename="prediction_results.csv"):
    """Create CSV file from prediction results"""
    if prediction_data and 'predictions' in prediction_data:
        print(f"\n📊 Creating CSV file {filename}...")
        try:
            predictions = prediction_data['predictions']
            df = pd.DataFrame(predictions)
            df.to_csv(filename, index=False)
            print(f"✅ CSV file created with {len(df)} rows")
        except Exception as e:
            print(f"❌ Failed to create CSV: {str(e)}")

def main():
    """Main test function"""
    print("🚀 Starting API Test Suite for Single Station Power Prediction")
    print("=" * 60)
    
    # Test 1: Health Check
    if not test_health_check():
        print("❌ Health check failed. Make sure the API server is running.")
        return
    
    # Test 2: List Stations
    stations = test_list_stations()
    
    # Test 3: Load Model
    if not test_load_model(TEST_STATION_ID):
        print(f"❌ Failed to load model for station {TEST_STATION_ID}")
        print("   Make sure the model exists. You may need to train it first.")
        return
    
    # Test 4: Generate Test Data
    test_data = generate_test_data(TEST_STATION_ID, days=7)
    
    # Test 5: Make Prediction
    prediction_result = test_prediction(TEST_STATION_ID, test_data)
    
    if prediction_result:
        # Save results
        save_prediction_results(prediction_result)
        create_prediction_csv(prediction_result)
        
        print("\n🎉 All tests completed successfully!")
        print("📊 Prediction Summary:")
        print(f"   Station: {prediction_result['station_id']}")
        print(f"   Prediction points: {prediction_result['total_points']}")
        print(f"   Time range: {prediction_result['prediction_start']} to {prediction_result['prediction_end']}")
    else:
        print("\n❌ Prediction test failed")
    
    print("=" * 60)
    print("🏁 Test suite completed")

if __name__ == "__main__":
    main()