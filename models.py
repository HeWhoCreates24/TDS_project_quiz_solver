"""
Data models and tracking classes for quiz solving
"""
import time
from typing import Dict, Any


class QuizAttempt:
    """Tracks a single attempt at solving a quiz"""
    def __init__(self, attempt_number: int):
        self.attempt_number = attempt_number
        self.start_time = time.time()
        self.end_time = None
        self.plan = None
        self.answer = None
        self.submission_response = None
        self.correct = None
        self.error = None
        self.artifacts = {}
        self.execution_log = []
        
    def finish(self):
        self.end_time = time.time()
    
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "plan_summary": {
                "tasks_count": len(self.plan.get("tasks", [])) if self.plan else 0,
                "final_answer_spec": self.plan.get("final_answer_spec") if self.plan else None
            },
            "answer": str(self.answer)[:200] if self.answer else None,
            "correct": self.correct,
            "error": self.error,
            "artifacts_count": len(self.artifacts),
            "execution_log_lines": len(self.execution_log)
        }


class QuizRun:
    """Tracks all attempts for a single quiz URL with time-based retry window"""
    def __init__(self, quiz_url: str):
        self.quiz_url = quiz_url
        self.first_attempt_time = None
        self.attempts = []
        self.current_attempt = None
        
    def start_attempt(self) -> QuizAttempt:
        if not self.first_attempt_time:
            self.first_attempt_time = time.time()
        self.current_attempt = QuizAttempt(len(self.attempts) + 1)
        self.attempts.append(self.current_attempt)
        return self.current_attempt
    
    def finish_current_attempt(self):
        if self.current_attempt:
            self.current_attempt.finish()
    
    def elapsed_time_since_first(self) -> float:
        """Get elapsed time since first attempt started"""
        if self.first_attempt_time:
            return time.time() - self.first_attempt_time
        return 0
    
    def can_retry(self, max_time: int = 150) -> bool:
        """Check if we can retry within time window"""
        return self.elapsed_time_since_first() < max_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quiz_url": self.quiz_url,
            "total_attempts": len(self.attempts),
            "total_time": self.elapsed_time_since_first(),
            "attempts": [attempt.to_dict() for attempt in self.attempts]
        }
