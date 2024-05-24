import mlrun
def load_project():
    return mlrun.get_or_create_project("flipkart-review1", context="./", user_project=True)