from lic.tasks.actions import TaskActions
from lic.tasks.code import TaskCode
from lic.tasks.data2text import TaskData2Text
from lic.tasks.database import TaskDatabase
from lic.tasks.math import TaskMath
from lic.tasks.summary import TaskSummary
from lic.tasks.translation import TaskTranslation


def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name.startswith("database"):
        return TaskDatabase(**kwargs)
    elif task_name == "code":
        return TaskCode(**kwargs)
    elif task_name == "translation":
        return TaskTranslation(**kwargs)
    elif task_name == "summary":
        return TaskSummary(**kwargs)
    elif task_name == "data2text":
        return TaskData2Text(**kwargs)
    elif task_name == "math":
        return TaskMath(**kwargs)
    elif task_name.startswith("actions"):
        return TaskActions(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported")


if __name__ == "__main__":
    task = get_task("data2text")
    print(len(task.get_samples()))
