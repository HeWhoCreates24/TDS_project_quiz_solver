# Test Quiz Endpoints

Local testing endpoints for the quiz solver without needing external quiz URLs.

## Available Test Endpoints

### 1. Get Test Quiz
```
GET /test-quiz/{quiz_type}
```

**Quiz Types:**
- `simple` - Returns a literal value quiz
- `calculation` - Requires basic calculation
- `scraping` - Requires downloading and parsing CSV
- `data` - Requires data filtering and statistics

**Example:**
```bash
curl http://localhost:8080/test-quiz/simple
```

**Response:**
```json
{
  "text": "The answer to this quiz is: literal_test_value",
  "links": [],
  "audio_sources": [],
  "video_sources": [],
  "image_sources": [],
  "code_blocks": []
}
```

### 2. Get Test Data Files
```
GET /test-data/{filename}
```

**Available Files:**
- `sample.csv` - 5 rows with id and amount columns
- `dataset.csv` - 20 rows with id, value, and score columns

**Example:**
```bash
curl http://localhost:8080/test-data/sample.csv
```

### 3. Submit Quiz Answer
```
POST /test-quiz/{quiz_type}/submit
```

**Request Body:**
```json
{
  "answer": "your_answer_here"
}
```

**Response:**
```json
{
  "correct": true,
  "submitted_answer": "your_answer_here",
  "message": "Correct!"
}
```

## Expected Answers

| Quiz Type    | Expected Answer | Description                          |
|--------------|-----------------|--------------------------------------|
| simple       | literal_test_value | Literal string from quiz text     |
| calculation  | 55              | Sum of 1 to 10                      |
| scraping     | 850             | Sum of 'amount' column in sample.csv|
| data         | 79.36           | Mean of 'score' where value >= 100  |

## Testing with Python Script

### Test All Quiz Types
```bash
python test_local_quiz.py all
```

### Test Specific Quiz Type
```bash
python test_local_quiz.py simple
python test_local_quiz.py calculation
python test_local_quiz.py scraping
python test_local_quiz.py data
```

### Test Endpoints Only
```bash
python test_local_quiz.py endpoints
```

### Test Default (Simple Quiz)
```bash
python test_local_quiz.py
```

## Manual Testing with cURL

### Test Simple Quiz Flow

1. **Get the quiz:**
```bash
curl http://localhost:8080/test-quiz/simple
```

2. **Solve it with the pipeline:**
```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "your_secret_key",
    "url": "http://localhost:8080/test-quiz/simple"
  }'
```

3. **Submit answer manually (optional):**
```bash
curl -X POST http://localhost:8080/test-quiz/simple/submit \
  -H "Content-Type: application/json" \
  -d '{"answer": "literal_test_value"}'
```

## Use Cases

### 1. Test Completion Detection Optimization
Use the test endpoints to verify that the completion detection is working correctly without making external API calls.

### 2. Benchmark Performance
Test different quiz types to measure performance improvements from optimizations.

### 3. Debug Issues
Use simple, controlled quizzes to isolate and debug specific issues without external dependencies.

### 4. CI/CD Testing
Integrate test endpoints into automated testing pipelines for regression testing.

## Notes

- All test endpoints run locally on port 8080
- No authentication required for test endpoints
- Test data is generated in-memory (no file I/O)
- Perfect for development and testing without internet connection
