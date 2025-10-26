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
        
    print("Transcription:", transcription.text)
    print("Video Length:", video_length.text)
    print("Music Theme:", music_theme.text)
    print("Company Name:", company_name.text)
    print("Company Industry:", company_industry.text)

    # Clean up - delete the uploaded file
    client.files.delete(name=myfile.name)
    print("Cleanup complete.")

# Test the correct file upload syntax
client = genai.Client(api_key="AIzaSyDQUqGULAZUdWB3CNyiKYwyafB83o0K9uU")

# Individual upload
# video_path = f"/Users/navitraa/ad-intelligence/inputs/videos/v0001.mp4"
# transcribe_video(video_path)

# Batch upload 
for i in range(1, 2):
    video_path = f"/Users/navitraa/ad-intelligence/inputs/videos/v{i:04d}.mp4"
    transcribe_video(video_path)

    output = pd.DataFrame({
        'video_file': Path(video_path).name,
        'transcription': transcription.text.strip(),
        'video_length': video_length.text.strip(),
        'music_theme': music_theme.text.strip(),
        'company_name': company_name.text.strip(),
        'company_industry': company_industry.text.strip()
    })

    output.to_excel("/Users/navitraa/ad-intelligence/outputs/video_features.xlsx", index=False)



