from src.api.v1.generators.manager import ProcessManager
from src.api.v1.jobs.schemas import JobSchema
from src.api.v1.jobs.user_inputs import JobUserInput
from src.core.config import Config

from .repositories import JobRepo


class JobService:
    job_repo: JobRepo
    manager: ProcessManager

    def __init__(self, job_repo: JobRepo, manager: ProcessManager):
        self.job_repo = job_repo
        self.manager = manager

    async def create_job(self, config: Config, input: JobUserInput) -> JobSchema:
        job = await self.job_repo.create(config.images_path, input)
        assert job.id is not None
        await self.manager.send_signal_new_job(job.id)
        return job
