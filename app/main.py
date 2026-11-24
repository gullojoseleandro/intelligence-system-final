import hmac
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.jobs import JobStore


class TaskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class ApprovalRequest(BaseModel):
    approved: bool


class TaskAccepted(BaseModel):
    job_id: str
    status: str


class TaskView(BaseModel):
    job_id: str
    status: str
    answer: str | None = None
    error: str | None = None
    approval: dict | None = None


def create_app(redis_client: Redis | None = None, admin_token: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv()
        token = admin_token or os.getenv("ADMIN_TOKEN", "")
        if not token or token == "replace-with-a-long-random-secret":
            raise RuntimeError("Set ADMIN_TOKEN before starting the API")
        client = redis_client if redis_client is not None else Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
        )
        try:
            await client.ping()
            app.state.store = JobStore(client)
            app.state.admin_token = token
            yield
        finally:
            if redis_client is None:
                await client.aclose()

    app = FastAPI(title="Multi-Agent Job API", version="0.1.0", lifespan=lifespan)

    @app.post("/tasks", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def create_task(body: TaskRequest) -> TaskAccepted:
        job_id = await app.state.store.create(body.query)
        return TaskAccepted(job_id=job_id, status="PENDING")

    @app.get("/tasks/{job_id}", response_model=TaskView)
    async def get_task(job_id: str) -> TaskView:
        job = await app.state.store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskView(
            job_id=job_id, status=job["status"], answer=job.get("answer"), error=job.get("error"),
            approval=json.loads(job["approval"]) if "approval" in job else None,
        )

    @app.post("/tasks/{job_id}/approve", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def approve_task(body: ApprovalRequest, job_id: str, x_admin_token: str | None = Header(default=None)) -> TaskAccepted:
        if x_admin_token is None or not hmac.compare_digest(x_admin_token, app.state.admin_token):
            raise HTTPException(status_code=401, detail="Invalid admin token")
        outcome = await app.state.store.approve(job_id, body.approved)
        if outcome == "missing":
            raise HTTPException(status_code=404, detail="Task not found")
        if outcome != "accepted":
            raise HTTPException(status_code=409, detail="Task is not waiting for approval")
        return TaskAccepted(job_id=job_id, status="PENDING")

    @app.get("/health")
    async def health() -> dict[str, str]:
        await app.state.store.redis.ping()
        return {"status": "ok"}

    return app


app = create_app()
