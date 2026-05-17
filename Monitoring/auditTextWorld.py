from monitor import Monitor
from utils import *
from ltl import LTLFormula
import ast
import os
import argparse

def textworldLTL():
    propositions = ["examine", "take", "drop", "knife", "chop", "slice", "go", "dice"]

    examine = LTLFormula.atom("examine")
    take = LTLFormula.atom("take")
    drop = LTLFormula.atom("drop")
    chop = LTLFormula.atom("chop")
    dice = LTLFormula.atom("dice")
    knife = LTLFormula.atom("knife")
    slice = LTLFormula.atom("slice")
    go = LTLFormula.atom("go")
    ltl_obj1 = LTLFormula.always(LTLFormula.implies(
        take,
        LTLFormula.next(examine)
    ))
    ltl_obj3 = LTLFormula.always(LTLFormula.neg(drop))
    ltl_obj4 = LTLFormula.always(LTLFormula.neg(go))

    sharp_act = LTLFormula.or_(LTLFormula.or_(chop, slice), dice)
    next_sharp_act = LTLFormula.next(sharp_act)
    next2_sharp_act = LTLFormula.next(LTLFormula.next(sharp_act))
    next3_sharp_act = LTLFormula.next(LTLFormula.next(LTLFormula.next(sharp_act)))
    next4_sharp_act = LTLFormula.next(LTLFormula.next(LTLFormula.next(LTLFormula.next(sharp_act))))

    ltl_obj2 = LTLFormula.always(LTLFormula.implies(
        LTLFormula.and_(take, knife),
        LTLFormula.or_(LTLFormula.or_(next_sharp_act, next2_sharp_act), LTLFormula.or_(next3_sharp_act, next4_sharp_act))))

    return [ltl_obj1, ltl_obj2, ltl_obj3, ltl_obj4], propositions, []



def audit_with_LLM(model, file, constraints):
    prompt = read_file("./prompts/audit-prompt")
    prompt += constraints + "\n"
    prompt += file
    print("------------------")
    ans = prompt_model(prompt, model, "")
    return ans

def audit_with_ICL_LLM(model, file, constraints):
    prompt = read_file("./prompts/audit-prompt")
    prompt += constraints + "\n"
    prompt += "These are the actions of the agent you are auditing: " + file
    print("------------------")
    ans = prompt_model(prompt, model, "")
    return ans

def extract_labels(entry):
    ans = ""
    if entry['drop']:
        ans += "Dropping objects. "
    if entry['examine']:
        ans += "Examining objects. "
    if entry['take'] and entry['knife']:
        ans += "Taking a knife. "
    elif entry['take']:
        ans += "Taking objects. "
    if entry['chop'] or entry['slice'] or entry['dice']:
        ans += "Using the knife to chop, slice or dice the food. "
    if entry['go']:
        ans += "Going to another location. "
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
        if len(l) > 1:
            ans.append(" Action: " + action + "  --- [State Oracle]: " + l +  "\n")
        else:
            ans.append(" Action: " + action + " \n")

    return "".join(ans)


def audit_with_icl_and_labels(model, file, labels):
    prompt = read_file("./prompts/audit-prompt-textworld-labels")
    prompt += "These are the actions of the agent you are auditing along with the state oracles: "
    labels_and_actions = merge_action_and_labels(file, labels)
    prompt += labels_and_actions
    print(prompt)
    ans = prompt_model(prompt, model, "")
    return ans


def main(args):
    constraints = read_file("./tests/textworld.txt")

    output_file = "./outputs/audit-textworld/"
    test_directory = "./tests/textworld/"
    raw_output = "./outputs/audit-textworld-raw/"
    label_directory = "./outputs/textworld-label/"

    files = os.listdir(test_directory)
    for file_name in files:
        file_path = os.path.join(test_directory, file_name)
        if len(args.filter) > 1:
            if args.filter not in file_name:
                continue
        if os.path.isfile(file_path):
            lines_raw = (read_file(file_path))
            model_name = args.model
            model_name = model_name.replace("/", "")
            if args.ltl:
                print("LTL")
                lines = lines_raw.strip().split("\n")
                ltl_obj, propositions, permanent = textworldLTL()

                if args.labelfile:
                    label_path = os.path.join(label_directory, file_name)
                    labelfile = label_path[:-4] + "---" + args.labelfile + "--labels.txt"
                else:
                    labelfile = None
                print(labelfile, "###")
                monitor = Monitor(ltl_obj, [], propositions, log_file=output_file  +  file_name + args.ex, LLM_label=args.label, permanent_propositions=permanent, labelfile = labelfile)
                for i, l in enumerate(lines):
                    l = l.lower()
                    monitor.step(l, "", i + 1)
            elif args.llm:
                print("LLM")
                output_file = raw_output + file_name + "--llm--" + model_name + "--zero" + args.ex + ".txt"
                if os.path.exists(output_file):
                    print(output_file, "exists, skipping")
                    continue
                ans = audit_with_LLM(args.model, lines_raw, constraints)

                with open(output_file, "w") as file:
                    file.write(ans)
            elif args.llmlabel:
                print("LLM with Labels")
                label_file = label_directory + file_name[:-4] + "---" + args.func + "--labels.txt"
                print(label_file)
                output_file = raw_output  +  file_name + "--llm--" + model_name + args.ex + ".txt"
                if os.path.exists(output_file):
                    print(output_file, "exists, skipping")
                    continue
                ans = audit_with_icl_and_labels(args.model, lines_raw, label_file)
                with open(output_file, "w") as file:
                    file.write(ans)
            elif args.icl:
                print("ICL LLM")
                output_file = raw_output + file_name + "--llm--" + model_name + "--finalicl" + args.ex + ".txt"
                constraints = read_file("./tests/textworld-icl.txt")
                if os.path.exists(output_file):
                    print(output_file, "exists, skipping")
                    continue
                ans = audit_with_ICL_LLM(args.model, lines_raw, constraints)

                with open(output_file, "w") as file:
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

    parser.add_argument('--llmlabel', action='store_true',
                        help='LLM labels that were generated before')

    parser.add_argument('--icl', action='store_true',
                        help='ICL')

    parser.add_argument('--labelfile', type=str, default=None, help='label file')
    args = parser.parse_args()
    main(args)
