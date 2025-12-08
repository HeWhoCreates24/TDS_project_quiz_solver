"""
Parallel task execution engine
Analyzes dependencies and executes independent tasks concurrently

PHASE 3: Full parallelization with CPU-bound operations
- I/O-bound tasks: async parallel (download, fetch, scrape, API calls)
- CPU-bound tasks: thread pool parallel (parsing, statistics, dataframe ops)
- Mixed waves: sequential execution
- Shared state protection: locks for dataframe_registry
"""
import logging
import asyncio
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# Global lock for dataframe_registry access
_dataframe_lock = threading.Lock()

# Tool categorization for parallel execution
IO_BOUND_TOOLS = {
    'download_file', 'fetch_text', 'render_js_page', 
    'scrape_with_javascript', 'fetch_from_api',
    'extract_html_text'
}

CPU_BOUND_TOOLS = {
    'parse_csv', 'parse_excel', 'parse_json_file', 
    'parse_html_tables', 'parse_pdf_tables',
    'calculate_statistics', 'dataframe_ops',
    'transform_dataframe', 'aggregate_data', 'reshape_data',
    'filter_data', 'sort_data',
    'transcribe_audio', 'extract_audio_metadata',
    'clean_text', 'extract_from_pdf'
}

def get_dataframe_lock() -> threading.Lock:
    """Get the global dataframe registry lock for safe concurrent access"""
    return _dataframe_lock


def build_dependency_graph(tasks: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Analyze task dependencies and group tasks into parallel execution waves.
    Tasks in the same wave can run concurrently.
    
    Returns: List of waves, where each wave is a list of independent tasks
    """
    if not tasks:
        return []
    
    # Build dependency map: task_id -> set of artifact keys it depends on
    task_deps = {}
    task_produces = {}
    dataframe_producers = {}  # Track which tasks produce dataframes
    
    for task in tasks:
        task_id = task["id"]
        inputs = task.get("inputs", {})
        produces = task.get("produces", [])
        tool_name = task.get("tool_name", "")
        
        # Track what this task produces
        # Handle both old format (list of dicts) and new format (list of strings)
        if produces:
            if isinstance(produces, list):
                if len(produces) > 0 and isinstance(produces[0], dict):
                    # Old format: [{"key": "artifact_name", ...}]
                    task_produces[task_id] = {p["key"] for p in produces if isinstance(p, dict) and "key" in p}
                else:
                    # New format: ["artifact_name"] or mixed
                    task_produces[task_id] = {p if isinstance(p, str) else p.get("key", "") for p in produces if p}
            elif isinstance(produces, str):
                # Single string instead of list
                logger.warning(f"[PARALLEL] Task {task_id} has string 'produces': {produces}, expected list")
                task_produces[task_id] = {produces}
            else:
                logger.warning(f"[PARALLEL] Task {task_id} has unexpected 'produces' type: {type(produces)}")
                task_produces[task_id] = set()
        else:
            task_produces[task_id] = set()
        
        # Track dataframe-producing tasks
        if tool_name in ["parse_csv", "parse_excel", "parse_json_file", "parse_html_tables", "dataframe_ops"]:
            # These tools create/modify dataframes
            if "dataframe_key" in str(produces) or tool_name.startswith("parse_"):
                dataframe_producers[task_id] = tool_name
        
        # Track what artifacts this task depends on
        deps = set()
        for key, value in inputs.items():
            # Check if input references an artifact (not a literal URL/string)
            if isinstance(value, str):
                # Common artifact patterns: df_0, rendered_page_1, transcription, etc.
                if any(value.startswith(prefix) for prefix in ['df_', 'rendered_', 'transcribe_', 'parse_', 'download_', 'extracted_']):
                    deps.add(value)
                # Special handling for dataframe references (e.g., "dataframe": "df_1")
                elif key == "dataframe" and (value.startswith("df_") or "_" in value):
                    # This task depends on a dataframe - mark dependency on dataframe-producing tasks
                    deps.add(f"__dataframe_dependency_{value}")
        task_deps[task_id] = deps
    
    # Build waves - greedy algorithm
    waves = []
    remaining_tasks = tasks[:]
    available_artifacts = set(artifacts.keys())
    completed_dataframe_tasks = set()  # Track completed dataframe operations
    
    while remaining_tasks:
        # Find all tasks whose dependencies are satisfied
        wave = []
        for task in remaining_tasks:
            task_id = task["id"]
            deps = task_deps.get(task_id, set())
            
            # Check dataframe dependencies
            can_execute = True
            resolved_deps = set(deps)
            for dep in deps:
                if dep.startswith("__dataframe_dependency_"):
                    # This is a dataframe dependency - check if any dataframe-producing task completed
                    if not completed_dataframe_tasks:
                        can_execute = False
                        break
                    else:
                        # Dependency satisfied by previous dataframe task
                        resolved_deps.remove(dep)
            
            # Can execute if all dependencies are available
            if can_execute and resolved_deps.issubset(available_artifacts):
                wave.append(task)
        
        # No tasks can execute - circular dependency or missing artifact
        if not wave:
            logger.warning(f"[PARALLEL] Dependency deadlock detected. {len(remaining_tasks)} tasks cannot execute.")
            logger.warning(f"[PARALLEL] Available artifacts: {available_artifacts}")
            logger.warning(f"[PARALLEL] Remaining tasks: {[t['id'] for t in remaining_tasks]}")
            # Execute them sequentially as fallback
            wave = remaining_tasks[:]
        
        waves.append(wave)
        
        # Update available artifacts and remaining tasks
        for task in wave:
            task_id = task["id"]
            available_artifacts.update(task_produces.get(task_id, set()))
            if task_id in dataframe_producers:
                completed_dataframe_tasks.add(task_id)
            remaining_tasks.remove(task)
    
    logger.info(f"[PARALLEL] Built {len(waves)} execution waves from {len(tasks)} tasks")
    for i, wave in enumerate(waves):
        logger.info(f"[PARALLEL] Wave {i+1}: {len(wave)} tasks - {[t['id'] for t in wave]}")
    
    return waves


async def execute_tasks_parallel(tasks: List[Dict[str, Any]], artifacts: Dict[str, Any], iteration: int, task_executor_func) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute tasks in parallel waves based on dependencies.
    
    PHASE 3 PARALLELIZATION:
    - I/O-bound waves: async gather (concurrent HTTP/network operations)
    - CPU-bound waves: thread pool (concurrent parsing/computation)
    - Mixed waves: sequential execution
    
    Args:
        tasks: List of tasks to execute
        artifacts: Current artifacts dict (will be updated)
        iteration: Current iteration number
        task_executor_func: Async function to execute a single task
    
    Returns:
        tuple: (updated_artifacts, execution_log)
    """
    waves = build_dependency_graph(tasks, artifacts)
    execution_log = []
    updated_artifacts = {}
    
    # Thread pool for CPU-bound tasks (limit to 4 workers to avoid overwhelming system)
    thread_pool = ThreadPoolExecutor(max_workers=4)
    
    try:
        for wave_idx, wave in enumerate(waves):
            if len(wave) > 1:
                # Categorize tasks in this wave
                tool_types = {task["tool_name"] for task in wave}
                all_io_bound = all(task["tool_name"] in IO_BOUND_TOOLS for task in wave)
                all_cpu_bound = all(task["tool_name"] in CPU_BOUND_TOOLS for task in wave)
                
                if all_io_bound:
                    # PHASE 2: I/O-bound parallel execution (async)
                    logger.info(f"[PARALLEL] Wave {wave_idx + 1}/{len(waves)}: {len(wave)} I/O tasks (async parallel)")
                    wave_results = await asyncio.gather(
                        *[task_executor_func(task, {**artifacts, **updated_artifacts}, iteration) for task in wave],
                        return_exceptions=True
                    )
                    
                    # Process results
                    for task, task_result in zip(wave, wave_results):
                        if isinstance(task_result, Exception):
                            logger.error(f"Task {task['id']} failed: {task_result}")
                            execution_log.append({
                                "task_id": task["id"],
                                "status": "failed",
                                "tool": task["tool_name"],
                                "error": str(task_result),
                                "iteration": iteration
                            })
                            raise task_result
                        else:
                            execution_log.append(task_result["execution_record"])
                            updated_artifacts.update(task_result["artifacts"])
                
                elif all_cpu_bound:
                    # PHASE 3: CPU-bound parallel execution (thread pool)
                    logger.info(f"[PARALLEL] Wave {wave_idx + 1}/{len(waves)}: {len(wave)} CPU tasks (thread pool parallel)")
                    
                    # Run CPU-bound tasks in thread pool
                    loop = asyncio.get_event_loop()
                    futures = []
                    for task in wave:
                        # Wrap async executor in sync wrapper for thread pool
                        future = loop.run_in_executor(
                            thread_pool,
                            lambda t=task: asyncio.run(task_executor_func(t, {**artifacts, **updated_artifacts}, iteration))
                        )
                        futures.append(future)
                    
                    # Wait for all CPU tasks to complete
                    wave_results = await asyncio.gather(*futures, return_exceptions=True)
                    
                    # Process results
                    for task, task_result in zip(wave, wave_results):
                        if isinstance(task_result, Exception):
                            logger.error(f"Task {task['id']} failed: {task_result}")
                            execution_log.append({
                                "task_id": task["id"],
                                "status": "failed",
                                "tool": task["tool_name"],
                                "error": str(task_result),
                                "iteration": iteration
                            })
                            raise task_result
                        else:
                            execution_log.append(task_result["execution_record"])
                            updated_artifacts.update(task_result["artifacts"])
                
                else:
                    # Mixed types - execute sequentially for safety
                    logger.info(f"[PARALLEL] Wave {wave_idx + 1}/{len(waves)}: {len(wave)} mixed tasks (sequential)")
                    for task in wave:
                        task_result = await task_executor_func(task, {**artifacts, **updated_artifacts}, iteration)
                        execution_log.append(task_result["execution_record"])
                        updated_artifacts.update(task_result["artifacts"])
            else:
                # Single task in wave - execute directly
                logger.info(f"[PARALLEL] Wave {wave_idx + 1}/{len(waves)}: {len(wave)} task")
                task = wave[0]
                task_result = await task_executor_func(task, {**artifacts, **updated_artifacts}, iteration)
                execution_log.append(task_result["execution_record"])
                updated_artifacts.update(task_result["artifacts"])
    
    finally:
        # Clean up thread pool
        thread_pool.shutdown(wait=True)
    
    return updated_artifacts, execution_log
