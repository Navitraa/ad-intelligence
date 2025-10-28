from google import genai
import time
import pandas as pd
from pathlib import Path

def transcribe_video(video_path):
    print("Uploading video...")
    myfile = client.files.upload(file=video_path)

    # Wait for processing to complete
    print("Processing video...")
    while myfile.state == "PROCESSING":
        time.sleep(2)
        myfile = client.files.get(name=myfile.name)
        print(f"Status: {myfile.state}")

    if myfile.state == "FAILED":
        print(f"Video processing failed: {myfile.name}")
        exit(1)

    print("Video ready! Generating transcription...")

    # Test basic functionality
    transcription = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."]
    )
    video_length = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "What is the length of this video in seconds? Just give the number."]
    )
    music_theme = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "What is the theme of the music in this video? In one word. No additional details"]
    )
    company_name = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "What is the name of the company in this video? Just give the name."]
    )
    company_industry = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "What is the industry of the company in this video? Just give the industry - no brackets, no additional details"]
    )
        
    # Clean up - delete the uploaded file
    client.files.delete(name=myfile.name)
    print("Cleanup complete.")
    
    # Return extracted features as dictionary
    return {
        'video_file': Path(video_path).name,
        'transcription': transcription.text.strip(),
        'video_length': video_length.text.strip(),
        'music_theme': music_theme.text.strip(),
        'company_name': company_name.text.strip(),
        'company_industry': company_industry.text.strip()
    }

# Test the correct file upload syntax
client = genai.Client(api_key="NIL")

# Individual upload
# video_path = f"/Users/navitraa/ad-intelligence/inputs/videos/v0001.mp4"
# transcribe_video(video_path)

# Batch upload and Excel export
all_features = []

for i in range(1, 5):
    video_path = f"/Users/navitraa/ad-intelligence/inputs/videos/v{i:04d}.mp4"
    
    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Skipping {video_path} - file not found")
        continue
    
    print(f"\nProcessing video {i}/22: {Path(video_path).name}")
    
    try:
        features = transcribe_video(video_path)
        all_features.append(features)
        
        # Print current video features
        print(f"✅ {features['video_file']}")
        print(f"   Company: {features['company_name']}")
        print(f"   Industry: {features['company_industry']}")
        print(f"   Length: {features['video_length']} seconds")
        print(f"   Music: {features['music_theme']}")
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        continue

# Save to Excel
if all_features:
    df = pd.DataFrame(all_features)
    output_file = "/Users/navitraa/ad-intelligence/outputs/video_features.xlsx"
    
    # Create outputs directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"\n🎉 Successfully processed {len(all_features)} videos!")
    print(f"📊 Features saved to: {output_file}")
    print(f"\nColumns in Excel file:")
    for col in df.columns:
        print(f"  - {col}")
else:
    print("❌ No videos were successfully processed.")

