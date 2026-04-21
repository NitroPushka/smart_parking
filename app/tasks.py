from app import crud
from app.database import Session
from app.services.analysis import analyze_parking_image
from app.celery_app import celery_app


@celery_app.task(name="app.tasks.process_analysis")
def process_analysis(task_id: str):
    db = Session()
    try:
        db_task = crud.get_analysis_task(db, task_id)
        if not db_task:
            raise ValueError(f"Analysis task {task_id} was not found")

        crud.update_analysis_task_status(db, task_id, status="processing")

        result = analyze_parking_image(
            db=db,
            lot_id=db_task.lot_id,
            image_path=db_task.image_path,
        )
        crud.update_analysis_task_status(
            db,
            task_id,
            status="completed",
            result=result,
            error_message=None,
        )
        return {"task_id": task_id, "status": "completed", "result": result}
    except Exception as exc:
        crud.update_analysis_task_status(
            db,
            task_id,
            status="failed",
            result=None,
            error_message=str(exc),
        )
        raise
    finally:
        db.close()
