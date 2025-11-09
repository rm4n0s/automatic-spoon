from dataclasses import dataclass
from typing import LiteralString

AIMODEL_NOT_FOUND_ERROR = "aimodel-not-found"


@dataclass
class UserResponse:
    response: LiteralString
    status: int


user_error_responses = {
    AIMODEL_NOT_FOUND_ERROR: UserResponse(
        response="We couldn't find the model you were looking for.", status=404
    )
}
