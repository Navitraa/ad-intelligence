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
    call_to_action = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "What is the call to action in this video? Just give the call to action - no brackets, no additional details"]
    )
    brand_logo = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "How long did the brand logo appear in the video? Just give the duration in seconds."]
    )
    cuts = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "How many cuts were there in this video? Just give me the number with no extra details or text"]
    )
    product = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "What is the product in this video? Just give the product - no brackets, no additional details"]
    )
    product_display_count = client.models.generate_content(
        model="gemini-2.5-flash", contents=[myfile, "How many times was the product displayed in this video? Just give me the number with no extra details or text."]
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
        'company_industry': company_industry.text.strip(),
        'call_to_action': call_to_action.text.strip(),
        'brand_logo': brand_logo.text.strip(),
        'cuts': cuts.text.strip(),
        'product': product.text.strip(),
        'product_display_count': product_display_count.text.strip(),
    }

# Test the correct file upload syntax
client = genai.Client(api_key="NIL")

# Individual upload
# video_path = f"/Users/navitraa/ad-intelligence/inputs/videos/v0001.mp4"
# transcribe_video(video_path)

# Batch upload - collect all features first
all_features = []

for i in range(1, 3):
    video_path = f"/Users/navitraa/ad-intelligence/inputs/videos/v{i:04d}.mp4"
    features = transcribe_video(video_path)
    all_features.append(features)
    print(f"✅ Processed {features['video_file']}")

# Create single DataFrame with all videos
output = pd.DataFrame(all_features)

# Create outputs directory if it doesn't exist
Path("/Users/navitraa/ad-intelligence/outputs").mkdir(parents=True, exist_ok=True)

# Save all videos to Excel at once
output.to_excel("/Users/navitraa/ad-intelligence/outputs/video_features.xlsx", index=False)

print(f"\n🎉 Saved {len(all_features)} videos to Excel!")
