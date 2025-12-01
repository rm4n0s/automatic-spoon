from pytsterrors import TSTError

from src.api.v1.generators.manager import ProcessManager
from src.api.v1.generators.repositories import GeneratorRepo
from src.api.v1.jobs.schemas import JobSchema
from src.api.v1.jobs.user_inputs import JobUserInput
from src.core.config import Config
from src.core.enums import JobStatus

from .repositories import JobRepo


class JobService:
    job_repo: JobRepo
    generator_repo: GeneratorRepo
    manager: ProcessManager

    def __init__(
        self, generator_repo: GeneratorRepo, job_repo: JobRepo, manager: ProcessManager
    ):
        self.job_repo = job_repo
        self.generator_repo = generator_repo
        self.manager = manager

    async def _validate(self, input: JobUserInput) -> list[dict[str, str]]:
        res = []
        ok = await self.generator_repo.exists(id=input.generator_id)
        if not ok:
            res.append(
                {
                    "field": "generator_id",
                    "error": f"generator with id {input.generator_id} doesn't exist",
                }
            )

        return res

    async def create_job(self, config: Config, input: JobUserInput) -> JobSchema:
        job = await self.job_repo.create(config.images_path, input)
        assert job.id is not None
        await self.manager.send_signal_new_job(job.id)
        return job

    async def delete_job(self, job_id: int):
        job = await self.job_repo.get_or_none(job_id)
        if not job:
            raise TSTError(
                "job-not-found",
                f"Job with ID {job_id} not found",
                metadata={"status_code": 400},
            )

        if job.status == JobStatus.PROCESSING:
            raise TSTError(
                "job-gets-processed",
                f"Job with ID {job_id} is getting processed, can't be deleted yet",
                metadata={"status_code": 400},
            )
        await self.job_repo.delete(job_id)
