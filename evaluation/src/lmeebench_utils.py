import copy

def prepare_lmeebench_navigation_goals(scene_name, episode, all_navigation_goals):
    filtered_tasks = []
    filtered_qas = []
    qa_dict = {qa["object_id"]: qa for qa in episode["QAs"]}
    for goal_id in episode["Object_id"]:
        goal_inst_id = goal_id
        goal_type = "description"
        dset_same_cat_goals = [
            category_i
            for category_i in all_navigation_goals.values()
            for goal_item in category_i
            if goal_item.get("object_id") == goal_inst_id
        ]

        if not dset_same_cat_goals:
            print(f"goal {goal_inst_id} not found in navigation goals.")
            continue

        goal_inst = [x for x in dset_same_cat_goals[0] if x["object_id"] == goal_inst_id]
        if not goal_inst:
            continue

        if "lang_desc" in goal_inst[0]:
            if len(goal_inst[0]["lang_desc"].split(" ")) <= 55:
                filtered_tasks.append(goal_id)
                if goal_inst_id in qa_dict:
                    filtered_qas.append(qa_dict[goal_inst_id])
        else:
            filtered_tasks.append(goal_id)
            if goal_inst_id in qa_dict:
                filtered_qas.append(qa_dict[goal_inst_id])

    all_subtask_goals = []
    all_subtask_goal_types = []

    for idx, goal_id in enumerate(filtered_tasks):
        goal_type = "description" if idx % 2 == 0 else "image"

        dset_same_cat_goals = [
            category_i
            for category_i in all_navigation_goals.values()
            for goal_item in category_i
            if goal_item.get("object_id") == goal_id
        ]

        if not dset_same_cat_goals:
            print(f"goal {goal_id} not found in all_navigation_goals")
            continue

        children_categories = dset_same_cat_goals[0][0]["children_object_categories"]
        for child_category in children_categories:
            goal_key = f"{scene_name}.basis.glb_{child_category}"
            if goal_key not in all_navigation_goals:
                print(f"!!! {goal_key} not in navigation_goals")
                continue
            print(f"!!! {goal_key} added")
            dset_same_cat_goals[0].extend(all_navigation_goals[goal_key])

        assert len(dset_same_cat_goals) == 1, f"more than 1 goal categories for {goal_id}"

        goal_inst = [x for x in dset_same_cat_goals[0] if x["object_id"] == goal_id]

        if goal_type == "description" and "lang_desc" not in goal_inst[0]:
            print(f"{goal_id} has no lang_desc, switching goal_type to 'image'")
            goal_type = "image"

        all_subtask_goals.append(goal_inst)
        all_subtask_goal_types.append(goal_type)

    for idx, qa in enumerate(filtered_qas):
        goal_id = qa['object_id']
        dset_same_cat_goals = [
            category_i
            for category_i in all_navigation_goals.values()
            for goal_item in category_i
            if goal_item.get("object_id") == goal_id
        ]

        if not dset_same_cat_goals:
            continue

        goal_inst = [x for x in dset_same_cat_goals[0] if x["object_id"] == goal_id]
        if not goal_inst:
            continue

        goal_copy = copy.deepcopy(goal_inst[0])
        goal_copy["QA"] = qa

        all_subtask_goals.append([goal_copy])
        all_subtask_goal_types.append("QA")

    return all_subtask_goal_types, all_subtask_goals

