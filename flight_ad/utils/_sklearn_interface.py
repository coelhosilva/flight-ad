from numpy import array

__all__ = ['retrieve_partial_pipeline']


def retrieve_partial_pipeline(pipeline, up_to):
    if up_to in [step[0] for step in pipeline.steps]:
        last_step = array([up_to == step[0] for step in pipeline.steps]).argmax()
        partial_pipeline = pipeline[:last_step+1]
    else:
        raise(Exception("Informed step not available in the pipeline."))
    return partial_pipeline
