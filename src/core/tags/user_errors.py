from dataclasses import dataclass
from typing import LiteralString

AIMODEL_NOT_FOUND_ERROR = "aimodel-not-found"
ENGINE_NOT_FOUND_ERROR = "engine-not-found"
WRONG_INPUT = "wrong-input"


@dataclass
class UserResponse:
    response: LiteralString
    status: int


user_error_responses = {
    AIMODEL_NOT_FOUND_ERROR: UserResponse(
        response="We couldn't find the AI model you were looking for.", status=404
    ),
    ENGINE_NOT_FOUND_ERROR: UserResponse(
        response="We couldn't find the engine you were looking for.", status=404
    ),
    WRONG_INPUT: UserResponse(response="The input is wrong", status=400),
}
