"""
Test quiz endpoints for local testing
Provides mock quizzes and data for comprehensive testing
"""
import pandas as pd
import numpy as np
import json
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.responses import Response, HTMLResponse

router = APIRouter()

# ========== TEST QUIZ TEMPLATES ==========

@router.get("/test-quiz/{quiz_type}")
async def test_quiz(quiz_type: str):
    """
    Test quiz endpoints for local testing
    Comprehensive 13-quiz chain testing all major tool categories
    """
    quiz_templates = {
        "literal": {
            "text": "Return the literal string: literal_test_value",
            "links": [],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/literal/submit",
            "origin_url": "http://localhost:8080/test-quiz/literal"
        },
        "compute": {
            "text": "Calculate the sum of all numbers from 1 to 10.",
            "links": [],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/compute/submit",
            "origin_url": "http://localhost:8080/test-quiz/compute"
        },
        "web_api": {
            "text": "Fetch data from the API endpoint and extract the 'secret_code' field from the JSON response.",
            "links": ["http://localhost:8080/test-api/config"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/web_api/submit",
            "origin_url": "http://localhost:8080/test-quiz/web_api"
        },
        "text_extract": {
            "text": "Fetch the text from the URL and extract all email addresses. Return the count of unique emails found.",
            "links": ["http://localhost:8080/test-data/sample_text.txt"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/text_extract/submit",
            "origin_url": "http://localhost:8080/test-quiz/text_extract"
        },
        "file_parse": {
            "text": "Download and parse the JSON file. Find the product with the highest 'price' and return its 'id'.",
            "links": ["http://localhost:8080/test-data/products.json"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/file_parse/submit",
            "origin_url": "http://localhost:8080/test-quiz/file_parse"
        },
        "multimedia": {
            "text": "Download and transcribe the audio file. The transcription contains a number - return that number.",
            "links": ["http://localhost:8080/test-data/sample_audio.mp3"],
            "audio_sources": ["http://localhost:8080/test-data/sample_audio.mp3"],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/multimedia/submit",
            "origin_url": "http://localhost:8080/test-quiz/multimedia"
        },
        "data_analysis": {
            "text": "Download the CSV file, filter rows where 'value' >= 100, and calculate the mean of the 'score' column.",
            "links": ["http://localhost:8080/test-data/dataset.csv"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/data_analysis/submit",
            "origin_url": "http://localhost:8080/test-quiz/data_analysis"
        },
        "ml_challenge": {
            "text": "Download the training CSV file, build a linear regression model to predict 'y' from 'x', then predict y when x=50.",
            "links": ["http://localhost:8080/test-data/regression_data.csv"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/ml_challenge/submit",
            "origin_url": "http://localhost:8080/test-quiz/ml_challenge"
        },
        "js_render": {
            "text": "Render the JavaScript page and extract the dynamically generated secret code from the rendered content.",
            "links": ["http://localhost:8080/test-page/dynamic"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/js_render/submit",
            "origin_url": "http://localhost:8080/test-quiz/js_render"
        },
        "pdf_parse": {
            "text": "Download the PDF file and extract data from the table. Calculate the sum of the 'amount' column.",
            "links": ["http://localhost:8080/test-data/sample_table.pdf"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/pdf_parse/submit",
            "origin_url": "http://localhost:8080/test-quiz/pdf_parse"
        },
        "vision": {
            "text": "Download the image and use OCR to extract the text. The image contains a number - return that number.",
            "links": ["http://localhost:8080/test-data/sample_image.png"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": ["http://localhost:8080/test-data/sample_image.png"],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/vision/submit",
            "origin_url": "http://localhost:8080/test-quiz/vision"
        },
        "transform": {
            "text": "Download the CSV file and pivot the data: use 'category' as index, 'month' as columns, and 'sales' as values. Return the sum of all values in the 'January' column.",
            "links": ["http://localhost:8080/test-data/sales_data.csv"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/transform/submit",
            "origin_url": "http://localhost:8080/test-quiz/transform"
        },
        "visualize": {
            "text": "Download the CSV file, create a bar chart showing 'category' vs 'total_sales', and save it. Return the number of categories in the chart.",
            "links": ["http://localhost:8080/test-data/category_sales.csv"],
            "audio_sources": [],
            "video_sources": [],
            "image_sources": [],
            "code_blocks": [],
            "submit_url": "http://localhost:8080/test-quiz/visualize/submit",
            "origin_url": "http://localhost:8080/test-quiz/visualize"
        }
    }
    
    if quiz_type not in quiz_templates:
        raise HTTPException(
            status_code=404, 
            detail=f"Quiz type '{quiz_type}' not found. Available: {list(quiz_templates.keys())}"
        )
    
    # Get quiz data
    quiz_data = quiz_templates[quiz_type]
    
    # Generate HTML with images, audio, etc.
    html_parts = [
        "<html><head><title>Test Quiz</title></head><body>",
        f"<p>{quiz_data['text']}</p>",
        f"<p>Submit your answer to: {quiz_data['submit_url']}</p>"
    ]
    
    # Add links
    for link in quiz_data.get('links', []):
        html_parts.append(f'<a href="{link}">{link}</a><br/>')
    
    # Add images
    for img_src in quiz_data.get('image_sources', []):
        html_parts.append(f'<img src="{img_src}" alt="Quiz Image"/><br/>')
    
    # Add audio
    for audio_src in quiz_data.get('audio_sources', []):
        html_parts.append(f'<audio src="{audio_src}" controls></audio><br/>')
    
    # Add video
    for video_src in quiz_data.get('video_sources', []):
        html_parts.append(f'<video src="{video_src}" controls></video><br/>')
    
    # Add code blocks
    for code in quiz_data.get('code_blocks', []):
        html_parts.append(f'<pre>{code}</pre>')
    
    html_parts.append("</body></html>")
    
    return HTMLResponse(content="\n".join(html_parts))


# ========== TEST DATA ENDPOINTS ==========

@router.get("/test-data/{filename}")
async def test_data(filename: str):
    """
    Serve test data files for quiz testing
    """
    # CSV files
    if filename == "sample.csv":
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "amount": [100, 250, 75, 300, 125]
        })
        return Response(content=df.to_csv(index=False), media_type="text/csv")
    
    elif filename == "dataset.csv":
        df = pd.DataFrame({
            "id": range(1, 21),
            "value": [50, 120, 80, 150, 90, 200, 60, 110, 95, 180, 
                     70, 130, 85, 160, 100, 140, 75, 190, 105, 170],
            "score": [45, 78, 62, 89, 55, 92, 48, 71, 58, 85,
                     52, 76, 60, 83, 68, 79, 54, 88, 70, 81]
        })
        return Response(content=df.to_csv(index=False), media_type="text/csv")
    
    elif filename == "regression_data.csv":
        # Simple linear relationship: y = 2*x + 10
        np.random.seed(42)
        x = np.linspace(0, 100, 50)
        y = 2 * x + 10 + np.random.normal(0, 5, 50)  # Add some noise
        df = pd.DataFrame({"x": x, "y": y})
        return Response(content=df.to_csv(index=False), media_type="text/csv")
    
    elif filename == "sales_data.csv":
        df = pd.DataFrame({
            "category": ["Electronics", "Electronics", "Clothing", "Clothing", "Food", "Food"],
            "month": ["January", "February", "January", "February", "January", "February"],
            "sales": [500, 300, 200, 150, 100, 80]
        })
        return Response(content=df.to_csv(index=False), media_type="text/csv")
    
    elif filename == "category_sales.csv":
        df = pd.DataFrame({
            "category": ["Electronics", "Clothing", "Food", "Books", "Toys"],
            "total_sales": [1200, 800, 450, 350, 600]
        })
        return Response(content=df.to_csv(index=False), media_type="text/csv")
    
    # Text files
    elif filename == "sample_text.txt":
        text = """Contact us at support@example.com for help.
Our sales team can be reached at sales@company.org.
For technical issues, email tech@example.com.
Marketing inquiries: marketing@example.com
Duplicate contact: support@example.com"""
        return Response(content=text, media_type="text/plain")
    
    # JSON files
    elif filename == "products.json":
        data = {
            "products": [
                {"id": "P001", "name": "Widget", "price": 29.99},
                {"id": "P002", "name": "Gadget", "price": 149.99},
                {"id": "P003", "name": "Doohickey", "price": 79.50},
                {"id": "P004", "name": "Thingamajig", "price": 199.99},
                {"id": "P005", "name": "Whatchamacallit", "price": 49.99}
            ]
        }
        return Response(content=json.dumps(data, indent=2), media_type="application/json")
    
    # Mock files
    elif filename == "sample_audio.mp3":
        return Response(content=b"MOCK_AUDIO_THE_SECRET_NUMBER_IS_42", media_type="audio/mpeg")
    
    elif filename == "sample_table.pdf":
        # Serve real PDF file
        import os
        pdf_path = os.path.join(os.path.dirname(__file__), "test-data", "sample_table.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            return Response(content=pdf_content, media_type="application/pdf")
        else:
            raise HTTPException(status_code=404, detail="PDF file not found")
    
    elif filename == "sample_image.png":
        # Serve real image file
        import os
        image_path = os.path.join(os.path.dirname(__file__), "test-data", "sample_image.png")
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_content = f.read()
            return Response(content=image_content, media_type="image/png")
        else:
            # Fallback to mock if file doesn't exist
            return Response(content=b"MOCK_IMAGE_WITH_TEXT_2048", media_type="image/png")
    
    else:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")


# ========== TEST QUIZ SUBMISSION ==========

@router.post("/test-quiz/{quiz_type}/submit")
async def submit_test_quiz(quiz_type: str, request: Request):
    """
    Submit answer for test quiz and validate
    """
    body = await request.json()
    answer = body.get("answer")
    
    base_url = "http://localhost:8080"
    
    # Comprehensive 13-quiz chain: easy -> hard, testing all major tool categories
    quiz_chain = {
        "literal": {
            "expected": "literal_test_value",
            "next": f"{base_url}/test-quiz/compute"
        },
        "compute": {
            "expected": 55,  # sum(1..10) = 55
            "next": f"{base_url}/test-quiz/web_api"
        },
        "web_api": {
            "expected": 7891,  # secret_code from API
            "next": f"{base_url}/test-quiz/text_extract"
        },
        "text_extract": {
            "expected": 4,  # 4 unique emails (support, sales, tech, marketing)
            "next": f"{base_url}/test-quiz/file_parse"
        },
        "file_parse": {
            "expected": "P004",  # Product with highest price (199.99)
            "next": f"{base_url}/test-quiz/multimedia"
        },
        "multimedia": {
            "expected": 42,  # Number from audio transcription
            "next": f"{base_url}/test-quiz/data_analysis"
        },
        "data_analysis": {
            "expected": 80.0,  # mean of scores where value >= 100
            "next": f"{base_url}/test-quiz/ml_challenge"
        },
        "ml_challenge": {
            "expected": 110.0,  # y = 2*50 + 10 = 110 (linear regression prediction)
            "tolerance": 5.0,  # Allow ±5 due to noise in training data
            "next": f"{base_url}/test-quiz/js_render"
        },
        "js_render": {
            "expected": 9876,  # Secret code from JS-rendered page
            "next": f"{base_url}/test-quiz/pdf_parse"
        },
        "pdf_parse": {
            "expected": 1250,  # Sum of amounts in PDF table
            "next": f"{base_url}/test-quiz/vision"
        },
        "vision": {
            "expected": 2048,  # Number extracted from image via OCR
            "next": f"{base_url}/test-quiz/transform"
        },
        "transform": {
            "expected": 800,  # Sum of January column after pivot
            "next": f"{base_url}/test-quiz/visualize"
        },
        "visualize": {
            "expected": 5,  # Number of categories in chart
            "next": None  # Last quiz in comprehensive chain
        }
    }
    
    if quiz_type not in quiz_chain:
        raise HTTPException(status_code=404, detail=f"Quiz type '{quiz_type}' not found")
    
    expected = quiz_chain[quiz_type]["expected"]
    tolerance = quiz_chain[quiz_type].get("tolerance", 0.01)  # Default small tolerance for floats
    next_url = quiz_chain[quiz_type]["next"]
    is_correct = False
    
    # Handle both string and numeric comparisons
    try:
        if isinstance(expected, (int, float)):
            # For numeric answers, use specified tolerance (or default)
            is_correct = abs(float(answer) - expected) < tolerance
        else:
            is_correct = str(answer).strip() == str(expected).strip()
    except (ValueError, TypeError):
        is_correct = False
    
    response = {
        "correct": is_correct,
        "submitted_answer": answer,
        "message": "Correct!" if is_correct else f"Incorrect. Expected: {expected}"
    }
    
    # Add next quiz URL only if answer is correct
    if is_correct and next_url:
        response["url"] = next_url
    
    return response


# ========== TEST API ENDPOINTS ==========

@router.get("/test-api/config")
async def test_api_config():
    """
    Mock API endpoint for web_api quiz
    Returns JSON with a secret code
    """
    return {
        "status": "success",
        "secret_code": 7891,
        "data": {
            "version": "1.0",
            "enabled": True
        }
    }


@router.get("/test-page/dynamic")
async def test_dynamic_page():
    """
    Mock JavaScript-rendered page for js_render quiz
    Returns HTML with JavaScript that generates dynamic content
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic Test Page</title>
    </head>
    <body>
        <h1>Test Page</h1>
        <div id="secret-container"></div>
        <script>
            // Simulate JavaScript-generated content
            document.addEventListener('DOMContentLoaded', function() {
                var secretCode = 9876;
                var container = document.getElementById('secret-container');
                container.innerHTML = '<p>The secret code is: <strong>' + secretCode + '</strong></p>';
            });
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)
