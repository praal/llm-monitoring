from auditScience import scienceworldLTL
from utils import *
import argparse
import os
from auditTextWorld import textworldLTL
from auditTruck import qual20_LTL


TEST_REGISTRY = {
    "scienceworld": {
        "ltl_func": scienceworldLTL,
        "test_directory": "./tests/scienceworld",
        "output_directory": "./outputs/science-label/",
        "truck_mode": False,
    },
    "textworld": {
        "ltl_func": textworldLTL,
        "test_directory": "./tests/textworld",
        "output_directory": "./outputs/textworld-label/",
        "truck_mode": False,
    },
    "qual20": {
        "ltl_func": qual20_LTL,
        "test_directory": "./tests/trucks/",
        "output_directory": "./outputs/qual20-label/",
        "truck_mode": True,
    },
}


class Labeler:
    def __init__(self, propositions=[], input_file = None, log_file=None, LLM_labeler=None, permanent_propositions = [], truck_mode = False):
        self.propositions = propositions
        self.LLM_labeler = LLM_labeler
        self.input = read_file(input_file)
        self.history = []
        self.log_file = log_file
        self.permanent_propositions = permanent_propositions
        self.state_history = dict()
        self.truck_areas = dict()
        self.truck_mode = truck_mode


        with open(self.log_file + "--labels.txt", "w") as file:
            file.write("")

        with open(self.log_file + "--prompts.txt", "w") as file:
            file.write("")
        self.log = []
        self.prompts = []
        if truck_mode:
            self.initialize_truck()

    def initialize_truck(self):
        self.state_history = dict()
        for a in ["a1", "a2", "a3"]:
            self.truck_areas[a] = []
        for p in self.permanent_propositions:
            if "free" in p:
                self.state_history[p] = True
            else:
                self.state_history[p] = False

    def truck_labeling_func(self, observation):

        state = dict()
        observation = observation.lower()
        for p in self.propositions:
            if "load package" in p:
                state[p] = False
                continue
            words = p.split(" ")
            all_true = True
            for w in words:
                if w not in observation:
                    all_true = False
            if all_true:
                state[p] = True
            else:
                state[p] = False

        all_drive = False
        for p in state.keys():
            if p in self.permanent_propositions and "drive" in p and state[p]:
                all_drive = True
                break

        for p in self.permanent_propositions:
            if "drive" in p and all_drive and (p not in state or not state[p]):
                state[p] = False
            elif p in self.state_history and self.state_history[p]:
                state[p] = True

            # Update truck area states
            if "free" not in p:
                continue

            for x in ["a1", "a2", "a3"]:
                if x not in state or not state[x] or x not in p:
                    continue
                if state["*load"]:
                    for item in range(1, 21):
                        pck = "package" + str(item)
                        if pck in observation and pck not in self.truck_areas[x]:
                            self.truck_areas[x].append(pck)
                if state["*unload"]:
                    for item in range(1, 21):
                        pck = "package" + str(item)
                        if pck in observation and pck in self.truck_areas[x]:
                            self.truck_areas[x].remove(pck)



        self.logger(state, [])


        for p in self.permanent_propositions:
            if "load" in p:
                x = p.split(" ")[-1]
                if x in observation and state["*load"]:
                    state[p] = True
            for x in ["a1", "a2", "a3"]:
                if x + " free before" in p:
                    if len(self.truck_areas[x]) == 0:
                        state[p] = True
                    else:
                        state[p] = False


       
        self.state_history = state.copy()

    def log_flush(self):

        with open(self.log_file + "--labels.txt", "a") as file:
            for l in self.log:
                file.write(str(l) + "\n")
        with open(self.log_file + "--prompts.txt", "a") as file:
            for l in self.prompts:
                file.write(str(l) + "\n")
        self.log = []
        self.prompts = []

    def logger(self, out, text):
        self.log.append(out)
        self.prompts.append(text)
        if len(self.log) >= 1:
            self.log_flush()

    def rule_based_labeling_func(self, observation):
        state = dict()
        for p in self.propositions:
            words = p.split(" ")
            all_true = True
            for w in words:
                if w not in observation:
                    all_true = False
            if all_true:
                state[p] = True
            else:
                state[p] = False
        self.logger(state, [])

    def update_history(self, observation):
        ob = observation.lower()
        if "load" in ob or "unload" in ob or "drive" in ob:
            self.history.append("Action: " + observation)

    def LLM_Truck_labeling_func_separate(self, raw_observation):
        out_dic = dict()
        prompts = []
        observation = raw_observation.replace("*", "")
        observation = observation.replace("-", " ")


        for p_raw in self.propositions:
            p = p_raw.replace("*", "")
            if not ("free before" in p or "drive to" in p):
                prompt = read_file("./prompts/labeling-prompt")
                prompt += "\n TEXT: " + observation + "\n TARGET: " + p
                ans = prompt_model(prompt, self.LLM_labeler, "Just output one answer as true or false.").lower()

                out_dic[p_raw] = False
                if 'true' in ans or 'yes' in ans:
                    out_dic[p_raw] = True
                if 'false' in ans or 'no' in ans or "can't" in ans or "I don't know" in ans[:-10]:
                    out_dic[p_raw] = False

                prompts.append(ans.replace("\n", " "))


        for p_raw in self.propositions:
            p = p_raw.replace("*", "")
            if not ("free before" in p or "drive to" in p):
                continue
            if "drive to" in p:
                if not out_dic["*deliver"]:
                    out_dic[p_raw] = False
                    continue
            if "free" in p:
                if not out_dic["*load"] and not out_dic["*unload"]:
                    out_dic[p_raw] = False
                    continue
                found = False
                for x in ["a1", "a2", "a3"]:
                    if x in p:
                        if not out_dic[x]:
                            found = True
                            out_dic[p_raw] = False
                if found:
                    continue
            if "drive to" in p:
                found = False
                for x in ["l1", "l2", "l3", "l4", "l5"]:
                    if x in p and x in observation:
                        found = True

                if not found:
                    out_dic[p_raw] = False
                    continue


            prompt = read_file("./prompts/labeling-truck-driveto")
            if "free before" in p:
                prompt = read_file("./prompts/labeling-truck-free")


            prompt += "History of actions: "
            if "drive to" in p:
                prompt += "The truck is at initial state." + "\n"
            else:
                prompt += "a1, a2 and a3 areas in the truck are initially free. " + "\n"

            if len(self.history) > 0:
                prompt += "\n".join(self.history)

            if "free before" not in p:
                prompt += observation

            prompt += "\n Proposition to evaluate (after executing the actions): "
            if "drive to" in p:
                prompt +=  p.replace("drive to-", "The truck is at location ") +  " after this action."
            elif "free before" in p:
                area_x = p[0:2]
                prompt += "Area " + area_x + " is free." + "There is no package loaded on area " + area_x + "."

            
            prompt += "\n Evaluate the propositions and output True or False:"



            ans = prompt_model(prompt, self.LLM_labeler, "Do not assume some other actions or observations have taken place. just output one answer as true or false.").lower()


            anssp = ans.lower().strip().split("\n")
            
            if 'true' in anssp[-1] or 'yes' in anssp[-1]:
                out_dic[p_raw] = True

            elif 'false' in anssp[-1] or 'no' in anssp[-1] or "can't" in anssp[-1] or "I don't know" in anssp[-1]:
                out_dic[p_raw] = False
            elif len(anssp) > 1 and 'false' in anssp[-1] or 'no' in anssp[-1] or "can't" in anssp[-1] or "I don't know" in anssp[-1]:
                out_dic[p_raw] = False
            else:
                out_dic[p_raw] = False

            prompts.append(ans.replace("\n", " "))
        self.update_history(observation)
        self.logger(out_dic, prompts)


    def LLM_labeling_func_separate(self, observation):
        out_dic = dict()
        prompts = []

        for p in self.propositions:
            prompt = read_file("./prompts/labeling-prompt")

            prompt += "\n TEXT: " + observation + "\n TARGET: " + p
            ans = prompt_model(prompt, self.LLM_labeler, "Just output one answer as true or false.").lower()

            if 'true' in ans or 'yes' in ans:
                out_dic[p] = True

            if 'false' in ans or 'no' in ans or "can't" in ans or "I don't know" in ans[:-10]:
                out_dic[p] = False

            prompts.append(ans.replace("\n", " "))
        self.logger(out_dic, prompts)

    def run(self, rule_based = False):
        lines = self.input.lower().split("\n")

        for i, l in enumerate(lines):

            if rule_based:
                if self.truck_mode:
                    self.truck_labeling_func(l)
                else:
                    self.rule_based_labeling_func(l)
                
            else:
                if self.truck_mode:
                    self.LLM_Truck_labeling_func_separate(l)
                else:
                    self.LLM_labeling_func_separate(l)
        self.log_flush()


def main(args):
    if args.test not in TEST_REGISTRY:
        raise ValueError(
            f"Unknown test '{args.test}'. Available tests: {sorted(TEST_REGISTRY.keys())}"
        )

    config = TEST_REGISTRY[args.test]
    _, propositions, permanent = config["ltl_func"]()
    test_directory = config["test_directory"]
    output_directory = config["output_directory"]
    truck_mode = config["truck_mode"]

    os.makedirs(output_directory, exist_ok=True)

    files = os.listdir(test_directory)
    for file_name in files:
        file_path = os.path.join(test_directory, file_name)
        if not os.path.isfile(file_path):
            continue

        out_file = os.path.join(
            output_directory,
            file_name[:-4] + "---" + args.model.replace("/", "") + args.ex,
        )


        if os.path.isfile(out_file + "--labels.txt"):
            print("File exists, skip:", out_file + "--labels.txt")

        labeler = Labeler(
            propositions,
            file_path,
            out_file,
            args.model,
            permanent_propositions=permanent,
            truck_mode=truck_mode,
        )

        labeler.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Labeler')
    parser.add_argument(
        '--test',
        type=str,
        default="scienceworld",
        choices=sorted(TEST_REGISTRY.keys()),
        help='which test/domain to label (selects LTL function and test directory)',
    )
    parser.add_argument('--model', type=str, default="", help='model')
    parser.add_argument('--ex', type=str, default="", help='external name')
    args = parser.parse_args()
    main(args)
