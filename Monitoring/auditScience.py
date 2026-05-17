from monitor import Monitor
from utils import *
from ltl import LTLFormula
import ast
import os
import argparse

def scienceworldLTL2():
    propositions = ["go", "look", "kitchen", "living room", "hallway", "art studio", "pick", "mix", "pour", "dunk", "yellow", "open", "around", "focus", "green color"]
    permanent = []
    go = LTLFormula.atom("go")
    look = LTLFormula.atom("look")
    kitchen = LTLFormula.atom("kitchen")
    living_room = LTLFormula.atom("living room")
    hallway = LTLFormula.atom("hallway")
    art = LTLFormula.atom("art studio")
    pick_atom = LTLFormula.atom("pick")
    mix_atom = LTLFormula.atom("mix")
    pour_atom = LTLFormula.atom("pour")
    dunk_atom = LTLFormula.atom("dunk")
    mix_pour_dunk = LTLFormula.or_(mix_atom, LTLFormula.or_(pour_atom, dunk_atom))
    open_atom = LTLFormula.atom("open")
    around = LTLFormula.atom("around")
    look_around = LTLFormula.and_(look, around)
    open_then_go = LTLFormula.always(LTLFormula.implies(open_atom, LTLFormula.next(go)))
    go_then_look = LTLFormula.always(LTLFormula.implies(go, LTLFormula.next(look_around)))
    look_twice = LTLFormula.always(LTLFormula.implies(look_around, LTLFormula.next(LTLFormula.neg(look_around))))
    go_allowed_locs = LTLFormula.always(LTLFormula.implies(go, LTLFormula.or_(kitchen, LTLFormula.or_(living_room, LTLFormula.or_(art, hallway)))))
    pick_then_look = LTLFormula.always(LTLFormula.implies(pick_atom, LTLFormula.next(look)))
    mix_pour_dunk_then_look = LTLFormula.always(LTLFormula.implies(mix_pour_dunk, LTLFormula.next(look)))
    yellow = LTLFormula.atom("yellow")
    pick_pour_dunk = LTLFormula.or_(pick_atom, LTLFormula.or_(pour_atom, dunk_atom))
    yellow_rule = LTLFormula.always(LTLFormula.implies(yellow, LTLFormula.neg(pick_pour_dunk)))

    focus_atom = LTLFormula.atom("focus")
    green = LTLFormula.atom("green color")
    focus_green = LTLFormula.always(LTLFormula.implies(focus_atom, green))
    return [open_then_go, go_then_look, look_twice, go_allowed_locs, pick_then_look, mix_pour_dunk_then_look, yellow_rule, focus_green], propositions, permanent


def scienceworldLTL():
    propositions = ["go", "look", "kitchen", "substance B", "box", "thermometer", "living room", "hallway", "open"]
    permanent = []
    go = LTLFormula.atom("go")
    look = LTLFormula.atom("look")
    kitchen = LTLFormula.atom("kitchen")
    box = LTLFormula.atom("box")
    thermometer = LTLFormula.atom("thermometer")
    substanceB = LTLFormula.atom("substance B")
    living_room = LTLFormula.atom("living room")
    hallway = LTLFormula.atom("hallway")
    open = LTLFormula.atom("open")

    open_then_go = LTLFormula.always(LTLFormula.implies(open, LTLFormula.next(go)))
    go_then_look = LTLFormula.always(LTLFormula.implies(go, LTLFormula.next(look)))
    look_twice = LTLFormula.always(LTLFormula.implies(look, LTLFormula.next(LTLFormula.neg(look))))
    go_allowed_locs = LTLFormula.always(LTLFormula.implies(go, LTLFormula.or_(kitchen, LTLFormula.or_(living_room, hallway))))
    go_twice = LTLFormula.always(LTLFormula.implies(go, LTLFormula.next(LTLFormula.neg(go))))
    box_rule = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(box, substanceB)), LTLFormula.and_(thermometer, substanceB))

    return [open_then_go, go_then_look, look_twice, go_allowed_locs, go_twice, box_rule], propositions, permanent


def audit_with_LLM(model, file, constraints):
    prompt = read_file("./prompts/audit-prompt")
    prompt += constraints + "\n"
    prompt += "These are the actions of the agent you are auditing: " + file
    ans = prompt_model(prompt, model, "")
    return ans

def audit_with_LLM_ICL(model, file, constraints):
    prompt = read_file("./prompts/audit-science-icl")
    prompt += "These are the actions of the agent you are auditing: " + file
    ans = prompt_model(prompt, model, "")
    return ans


def extract_labels(entry):
    ans = ""
    if entry['go']:
        if entry['kitchen']:
            ans += "Going to the kitchen. "
        elif entry['living room']:
            ans += "Going to the living room. "
        elif entry['hallway']:
            ans += "Going to the hallway. "
        else:
            ans += "Going to another location."
    if entry['look']:
        ans += "Looking around."
    if entry['open']:
        ans += "Opening. "
    if entry['box']:
        ans += "Interacting with the box. "
    if entry['thermometer']:
        ans += "Interacting with the thermometer. "
    if entry['substance B']:
        ans += "Interacting with substance B. "
    return ans


def merge_action_and_labels(file, labels_file):
    actions = file.strip().split("\n")
    labels = []
    ans = []
    with open(labels_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    entry = ast.literal_eval(line)
                    labels.append(entry)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")


    for i in range(len(actions)):
        action = actions[i]
        l = extract_labels(labels[i])
        if len(l) < 2:
            ans.append(" Action: " + action + "\n")
        else:
            ans.append(" Action: " + action + "  --- [State Oracle]: " + l +  "\n")

    return "".join(ans)


def audit_with_icl_and_labels(model, file, labels):
    prompt = read_file("./prompts/audit-prompt-science-labels")
    prompt += "These are the actions of the agent you are auditing along with the state oracles: "
    labels_and_actions = merge_action_and_labels(file, labels)
    prompt += labels_and_actions
    ans = prompt_model(prompt, model, "")
    return ans

def main(args):
    constraints = read_file("./tests/science-rules.txt")

    output_file_raw = "outputs/audit-science-raw/"
    output_file_clean = "outputs/audit-science/"
    test_directory = "./tests/scienceworld/"


    label_directory = "./outputs/science-label/"


    files = os.listdir(test_directory)
    for file_name in files:
        file_path = os.path.join(test_directory, file_name)
        if len(args.filter) > 1:
            if args.filter not in file_name:
                continue
        if os.path.isfile(file_path):
            print(file_name)
            lines_raw = (read_file(file_path))
            if args.ltl:
                print("LTL")
                lines = lines_raw.strip().split("\n")
                ltl_obj, propositions, permanent = scienceworldLTL()

                if args.labelfile:
                    label_path = os.path.join(label_directory, file_name)
                    labelfile = label_path[:-4] + "---" + args.labelfile + "--labels.txt"
                else:
                    labelfile = None
                print(labelfile, "###")
                if labelfile is not None and not os.path.exists(labelfile):
                    print("Label file does not exist:", labelfile)
                    continue
                if labelfile is not None:
                    label_lines = []
                    with open(labelfile, 'r') as file:
                        for line in file:
                            label_lines.append(line.strip())
                    if len(label_lines) > len(lines) + 2 or len(label_lines) < len(lines) -2:
                        print(len(label_lines), len(lines))
                        print("Label file length does not match action file length")
                        continue
                monitor = Monitor(ltl_obj, [], propositions, log_file=output_file_clean  +file_name + args.ex, LLM_label=args.label, permanent_propositions=permanent, labelfile = labelfile)
                for i, l in enumerate(lines):
                    l = l.lower()
                    monitor.step(l, "", i + 1)
            elif args.llm:
                print("LLM")
                ans = audit_with_LLM(args.model, lines_raw, constraints)
                model_name = args.model
                model_name = model_name.replace("/", "")
                with open(output_file_raw +  file_name + "--llm--" + model_name + args.ex + ".txt", "w") as file:
                    file.write(ans)

            elif args.icl:
                print("ICL")
                model_name = args.model
                model_name = model_name.replace("/", "")

                out_adr = output_file_raw +  file_name + "--llm--" + model_name + "--finalicl" + args.ex + ".txt"
                if os.path.exists(out_adr):
                    print("Already exists:", out_adr)
                    continue
                ans = audit_with_LLM_ICL(args.model, lines_raw, constraints)
                with open(out_adr, "w") as file:
                    file.write(ans)
            elif args.iclwithlabels:
                print("ICL with Labels")
                model_name = args.model
                model_name = model_name.replace("/", "")
                out_name = output_file_raw  + file_name + "--llm--" + model_name + "--iclwithlabels" + args.ex + ".txt"

                if os.path.exists(out_name):
                    print("Already exists:", out_name)
                    continue

                label_file = label_directory + file_name[:-4] + "---" + args.func + "--labels.txt"
                print(label_file)
                ans = audit_with_icl_and_labels(args.model, lines_raw, label_file)

                with open(out_name, "w") as file:
                    file.write(ans)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit an LLM')
    parser.add_argument('--model', type=str, default="meta-llama/llama-3.3-70b-instruct",
                        help='Model for audit')
    parser.add_argument('--llm', action='store_true',
                        help='LLM auditor')

    parser.add_argument('--ltl', action='store_true',
                        help='LTL auditor')


    parser.add_argument('--label', action='store_true',
                        help='LLM labeler')
    parser.add_argument('--filter', type=str, default="", help='file name filtering')

    parser.add_argument('--ex', type=str, default="", help='external naming')
    parser.add_argument('--func', type=str, default="", help='LLM that generated labels')


    parser.add_argument('--labelfile', type=str, default=None, help='label file')
    parser.add_argument('--icl', action='store_true')
    parser.add_argument('--iclwithlabels', action='store_true')
    args = parser.parse_args()
    main(args)
