class PerceptionError(RuntimeError):
    pass


class DependencyUnavailable(PerceptionError):
    pass


class ModelLoadError(PerceptionError):
    pass


class CudaOutOfMemory(PerceptionError):
    pass


def classify_exception(error: BaseException) -> PerceptionError:
    if isinstance(error, PerceptionError):
        return error
    text = str(error).lower()
    if isinstance(error, MemoryError) or "out of memory" in text:
        return CudaOutOfMemory(str(error))
    if any(token in text for token in ("weight", "checkpoint", "model load")):
        return ModelLoadError(str(error))
    return PerceptionError(str(error))
