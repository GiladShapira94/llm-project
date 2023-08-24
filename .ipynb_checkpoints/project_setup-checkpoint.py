import importlib

import mlrun


def assert_build():
    for module_name in [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "evaluate",
        "deepspeed",
        "mpi4py",
    ]:
        module = importlib.import_module(module_name)
        print(module.__version__)


def setup(project: mlrun.projects.MlrunProject):
    """
    Creating the project for this demo.
    :returns: a fully prepared project for this demo.
    """
    
    # Set the project git source
    source = project.get_param("source")
    if source:
        print(f"Project Source: {source}")
        project.set_source(project.get_param("source"), pull_at_runtime=True)

    # Set or build the default image:
    if project.get_param("default_image") is None:
        print("not-image")
        # print("Building image for the demo:")
        # image_builder = project.set_function(
        #     "project_setup.py",
        #     name="image-builder",
        #     handler="assert_build",
        #     kind="job",
        #     image="mlrun/ml-models-gpu",
        #     requirements=[
        #         "torch",
        #         "transformers[deepspeed]",
        #         "datasets",
        #         "accelerate",
        #         "evaluate",
        #         "mpi4py",
        #     ],
        # )
        # assert image_builder.deploy()
        # default_image = image_builder.spec.image
    project.set_default_image(project.get_param("default_image"))

    # Set the data collection function:
    transcribe_url = "https://raw.githubusercontent.com/GiladShapira94/functions/fix_fucntion_pii_qa/transcribe/function.yaml"
    transcribe_func = project.set_function(transcribe_url, name="transcribe")
    transcribe_func.apply(mlrun.auto_mount())
    transcribe_func.save()

    # Set the data preprocessing function:
    pii_recognizer_url = "https://raw.githubusercontent.com/GiladShapira94/functions/fix_fucntion_pii_qa/pii_recognizer/function.yaml"
    pii_recognizer_func = project.set_function(
        pii_recognizer_url, name="pii-recognizer"
    )

    # Set the training function:
    question_answering_url = "https://raw.githubusercontent.com/GiladShapira94/functions/fix_fucntion_pii_qa/question_answering/function.yaml"
    question_answering_func = project.set_function(
        question_answering_url, name="question-answering"
    )
    if project.get_param("with_gpu") == True:
        print("with-gpu")
        # question_answering_func.with_limits(gpus=1)
        # question_answering_func.save()
    postprocess_function = project.set_function(
        "./postprocess.py", kind="job", name="postprocess"
    )
    project.set_workflow("training_workflow", "./training_workflow.py")

    # Save and return the project:
    project.save()
    return project
