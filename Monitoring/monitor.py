import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import *
from ltl import LTLFormula
import ast

class Monitor:
    def __init__(self, formulas, state_formulas, propositions=[], log_file=None, auto_recovery=True, LLM_label = False, LLM_labeler="google/gemini-2.0-flash-lite-001", permanent_propositions = [], labelfile=None, truck_mode=False):
        self.initial_formulas = formulas
        self.current_formulas = formulas
        self.initial_state_formulas = state_formulas
        self.current_state_formulas = state_formulas
        self.history = []
        self.propositions = propositions
        self.permanent_propositions = permanent_propositions
        self.auto_recovery = auto_recovery
        self.LLM_label = LLM_label
        self.LLM_labeler = LLM_labeler
        self.truck_mode = truck_mode
        self.truck_areas = dict()

        self.labels = []
        self.step_cnt = 0
        self.labelfile = labelfile
        if self.labelfile is not None:
            with open(self.labelfile, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            entry = ast.literal_eval(line)
                            self.labels.append(entry)
                        except (SyntaxError, ValueError) as e:
                            print(f"Error parsing line: {line}")
                            print(f"Error: {e}")

        self.violation_step = -1
        self.satisfaction_step = -1

        self.violation_count = dict()
        for i in range(len(self.initial_formulas)):
            self.violation_count[i] = 0
        for i in range(len(self.initial_state_formulas)):
            self.violation_count[i + len(self.initial_formulas)] = 0

        self.log_file = log_file
        if self.log_file is not None:
            with open(self.log_file + "--monitor.txt", "w") as file:
                file.write("")
        self.log = []
        self.state_history = dict()
        self.satisfied = []
        self.initialize_truck()


    def __deepcopy__(self, memodict={}):
        new_monitor = Monitor(
            formulas=[f.__deepcopy__() for f in self.initial_formulas],
            state_formulas=[f.__deepcopy__() for f in self.initial_state_formulas],
            propositions=self.propositions.copy(),
            log_file=None,
            auto_recovery=self.auto_recovery,
            LLM_label=self.LLM_label,
            LLM_labeler=self.LLM_labeler,
            permanent_propositions=self.permanent_propositions.copy(),
            labelfile=self.labelfile,
            truck_mode=self.truck_mode
        )
        new_monitor.current_formulas = [f.__deepcopy__() for f in self.current_formulas]
        new_monitor.current_state_formulas = [f.__deepcopy__() for f in self.current_state_formulas]
        new_monitor.history = self.history.copy()
        new_monitor.step_cnt = self.step_cnt
        new_monitor.state_history = self.state_history.copy()
        new_monitor.satisfied = self.satisfied.copy()
        new_monitor.truck_areas = {k: v.copy() for k, v in self.truck_areas.items()}
        new_monitor.violation_count = self.violation_count.copy()
        new_monitor.violation_step = self.violation_step
        return new_monitor

    def log_flush(self):
        if self.log_file is None:
            return
        with open(self.log_file + "--monitor.txt", "a") as file:
            for l in self.log:
                file.write(str(l) + "\n")
        self.log = []

    def logger(self, history_entry):
        self.log.append(history_entry)
        if len(self.log) >= 1:
            self.log_flush()


    def initialize_truck(self):
        self.state_history = dict()
        for a in ["a1", "a2", "a3"]:
            self.truck_areas[a] = []
        for p in self.permanent_propositions:
            if "free" in p:
                self.state_history[p] = True
            else:
                self.state_history[p] = False


    def LLM_labeling_func_non_temporal(self, observation):
        out = dict()
        if self.labels is not None:
            out = self.labels[self.step_cnt]
            self.step_cnt += 1
        else:

            for p in self.propositions:
                prompt = "Given this observation: [Observation:" + observation + "] evaluate if the following action is true or false as the exact result of this observation, or if the following is an item, it is being used or not. [" + p + "]"
                ans = prompt_model(prompt, self.LLM_labeler, "Do not assume some other actions or observations have taken place. just output one answer as true or false.").lower()
                if 'true' in ans or 'yes' in ans:
                    out[p] = True
                if 'false' in ans or 'no' in ans or "can't" in ans or "I don't know" in ans:
                    out[p] = False

        self.state_history = out.copy()


        return out


    def truck_labeling_func(self, observation):
        state = dict()
        observation = observation.lower()
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

            for x in ["a1", "a2", "a3"]:
                if x + " free before" in p:
                    if len(self.truck_areas[x]) == 0:
                        state[p] = True
                    else:
                        state[p] = False


        # Update truck area states
        for x in ["a1", "a2", "a3"]:
            if x not in state or not state[x]:
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


        self.state_history = state.copy()
        for item in range(1, 21):
            pck = "package" + str(item)
            if state['*load'] and pck in observation:
                self.state_history['loaded ' + pck] = True
        return state


    def rule_based_labeling_func(self, observation):
        state = dict()
        observation = observation.lower()
        observation = observation.replace("$", "")
        observation = observation.replace(":", "")
        observation = observation.replace(",", "")
        observation = observation.replace(".", "")
        observation = observation.replace("-", "")
        observation = observation.strip()
        obs_words = observation.split(" ")
        for p in self.propositions:
            words = p.split(" ")
            all_true = True
            for w in words:
                if w not in obs_words:
                    all_true = False
            if all_true:
                state[p] = True
        for p in self.permanent_propositions:
            if p in self.state_history and self.state_history[p]:
                state[p] = True

        self.state_history = state.copy()
        return state

    def get_current_formulas(self):
        ans = self.current_formulas.copy()
        ans.extend(self.current_state_formulas)
        return ans





    def step(self, action, state_obs, step_index):
        if self.LLM_label or self.labelfile is not None:
            action_state = dict()
            obs_state = dict()
            if len(action) > 2:
                action_state = self.LLM_labeling_func_non_temporal(action)
            if len(state_obs) > 2:
                obs_state = self.LLM_labeling_func_non_temporal(state_obs)
        elif self.truck_mode:
            action_state = self.truck_labeling_func(action)
            obs_state = self.truck_labeling_func(state_obs)
        else:
            action_state = self.rule_based_labeling_func(action)
            if len(state_obs) > 2:
                obs_state = self.rule_based_labeling_func(state_obs)

        progressed_formulas = []
        for f in self.current_formulas:
            progressed_formulas.append(f.progress(action_state))

        obs_progressed_formulas = []
        for f in self.current_state_formulas:
            obs_progressed_formulas.append(f.progress(obs_state))


        history_entry = {
            'step': step_index,
            'action': action,
            'event': '',
            'vio-which': '',
            'satis-which': '',
            'action_state': action_state,
            #'obs_state': obs_state,
            'after': '',
            #'after_state': '',
            'total_vio': sum(self.violation_count.values()),
        }

        violated = 0

        for i,f in enumerate(progressed_formulas):
            if f.score() == -1:
                violated += 1
                self.violation_step = step_index
                self.violation_count[i] += 1
                history_entry['total_vio'] =  sum(self.violation_count.values())
                history_entry['event'] += '-violation'
                history_entry['vio-which'] += str(i + 1) + "-"
                if self.auto_recovery:
                    progressed_formulas[i] = self.initial_formulas[i]
                    history_entry['after'] += str(progressed_formulas[i])
                    history_entry['event'] += '-recovered'
            if f.score() == 1:
                if i in self.satisfied:
                    continue
                self.satisfied.append(i)
                history_entry['event'] += '-satisfaction'
                history_entry['satis-which'] += str(i+ 1) + "-"


        for i, f in enumerate(obs_progressed_formulas):
            if f.score() == -1:
                violated += 1
                self.violation_step = step_index
                self.violation_count[i + len(self.current_formulas)] += 1
                history_entry['total_vio'] = self.violation_count
                history_entry['event'] = 'violation'

                if self.auto_recovery:
                    obs_progressed_formulas[i] = self.initial_state_formulas[i]
                    history_entry['after_state'] += str(obs_progressed_formulas[i])
                    history_entry['event'] = 'violation-recovered'


        monitor_changed = violated

        self.history.append(history_entry)
        self.logger(history_entry)
        self.current_formulas = progressed_formulas
        self.current_state_formulas = obs_progressed_formulas


        return monitor_changed

    def get_verdict(self):
        all_satisfied = True
        for f in self.current_formulas:
            s = f.score()
            if  s == -1:
                return -1
            if s != 1:
                all_satisfied = False
        for f in self.current_state_formulas:
            s = f.score()
            if s == -1:
                return -1
            if s != 1:
                all_satisfied = False

        if all_satisfied:
            return 1
        return 0
