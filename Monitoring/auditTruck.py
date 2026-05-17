from monitor import Monitor
from utils import *
from ltl import LTLFormula
import ast
import os
import argparse

def qual20_LTL():
    propositions = ["*load", "*unload", "*deliver", "*drive", "a1", "a2", "a3", "package1", "package2", "package3",  "package5", "package6",  "package10", "package11",  "package15", "package16","package18", "package19", "package20", "l1",
                    "l2", "l3", "l4", "l5"]
    permanent = ["*drive to-l1", "*drive to-l2", "*drive to-l3", "*drive to-l4" ,"*drive to-l5" , "a1 free before", "a2 free before", "a3 free before"]

    permanent_package = ["loaded package1", "loaded package2", "loaded package3", "loaded package5", "loaded package6",
                         "loaded package10", "loaded package11", "loaded package15", "loaded package16",
                         "loaded package18", "loaded package19", "loaded package20"]

    permanent.extend(permanent_package)
    propositions.extend(permanent)
    a1_free_before = LTLFormula.atom("a1 free before")
    a2_free_before = LTLFormula.atom("a2 free before")
    a3_free_before = LTLFormula.atom("a3 free before")

    p1_loaded = LTLFormula.atom("loaded package1")
    p2_loaded = LTLFormula.atom("loaded package2")
    p3_loaded = LTLFormula.atom("loaded package3")
    p5_loaded = LTLFormula.atom("loaded package5")
    p6_loaded = LTLFormula.atom("loaded package6")
    p10_loaded = LTLFormula.atom("loaded package10")
    p11_loaded = LTLFormula.atom("loaded package11")
    p15_loaded = LTLFormula.atom("loaded package15")
    p16_loaded = LTLFormula.atom("loaded package16")
    p18_loaded = LTLFormula.atom("loaded package18")
    p19_loaded = LTLFormula.atom("loaded package19")
    p20_loaded = LTLFormula.atom("loaded package20")

    load_atom = LTLFormula.atom("*load")
    unload_atom = LTLFormula.atom("*unload")
    a1_atom = LTLFormula.atom("a1")
    a2_atom = LTLFormula.atom("a2")
    a3_atom = LTLFormula.atom("a3")

    load_in_a1 = LTLFormula.always(LTLFormula.implies(load_atom, a1_atom))
    load_in_a1_a2 = LTLFormula.always(LTLFormula.implies(load_atom, LTLFormula.or_(a2_atom, a1_atom)))
    load_in_a2_a2_a3 = LTLFormula.always(LTLFormula.implies(load_atom, LTLFormula.or_(a1_atom,LTLFormula.or_(a2_atom, a3_atom))))

    p1_atom = LTLFormula.atom("package1")
    p2_atom = LTLFormula.atom("package2")
    p3_atom = LTLFormula.atom("package3")
    p5_atom = LTLFormula.atom("package5")
    p6_atom = LTLFormula.atom("package6")
    p11_atom = LTLFormula.atom("package11")
    p10_atom = LTLFormula.atom("package10")
    p16_atom = LTLFormula.atom("package16")
    p15_atom = LTLFormula.atom("package15")
    p18_atom = LTLFormula.atom("package18")
    p19_atom = LTLFormula.atom("package19")
    p20_atom = LTLFormula.atom("package20")

    l1_atom = LTLFormula.atom("l1")
    l2_atom = LTLFormula.atom("l2")
    l3_atom = LTLFormula.atom("l3")
    l4_atom = LTLFormula.atom("l4")
    l5_atom = LTLFormula.atom("l5")

    deliver_atom = LTLFormula.atom("*deliver")
    load_1_atom = LTLFormula.and_(load_atom, p1_atom)
    load_2_atom = LTLFormula.and_(load_atom, p2_atom)
    load_3_atom = LTLFormula.and_(load_atom, p3_atom)
    load_5_atom = LTLFormula.and_(load_atom, p5_atom)
    load_6_atom = LTLFormula.and_(load_atom, p6_atom)
    load_10_atom = LTLFormula.and_(load_atom, p10_atom)
    load_11_atom = LTLFormula.and_(load_atom, p11_atom)
    load_15_atom = LTLFormula.and_(load_atom, p15_atom)
    load_16_atom = LTLFormula.and_(load_atom, p16_atom)
    load_18_atom = LTLFormula.and_(load_atom, p18_atom)
    load_19_atom = LTLFormula.and_(load_atom, p19_atom)
    load_20_atom = LTLFormula.and_(load_atom, p20_atom)

    unload_1_atom = LTLFormula.and_(unload_atom, p1_atom)
    unload_2_atom = LTLFormula.and_(unload_atom, p2_atom)
    unload_3_atom = LTLFormula.and_(unload_atom, p3_atom)
    unload_5_atom = LTLFormula.and_(unload_atom, p5_atom)
    unload_6_atom = LTLFormula.and_(unload_atom, p6_atom)
    unload_10_atom = LTLFormula.and_(unload_atom, p10_atom)
    unload_11_atom = LTLFormula.and_(unload_atom, p11_atom)
    unload_15_atom = LTLFormula.and_(unload_atom, p15_atom)
    unload_16_atom = LTLFormula.and_(unload_atom, p16_atom)
    unload_18_atom = LTLFormula.and_(unload_atom, p18_atom)
    unload_19_atom = LTLFormula.and_(unload_atom, p19_atom)
    unload_20_atom = LTLFormula.and_(unload_atom, p20_atom)

    p1_load_once = LTLFormula.always(
        LTLFormula.implies(load_1_atom, LTLFormula.neg(p1_loaded)))
    p2_load_once = LTLFormula.always(
        LTLFormula.implies(load_2_atom, LTLFormula.neg(p2_loaded)))
    p3_load_once = LTLFormula.always(
        LTLFormula.implies(load_3_atom, LTLFormula.neg(p3_loaded)))
    p5_load_once = LTLFormula.always(
        LTLFormula.implies(load_5_atom, LTLFormula.neg(p5_loaded)))
    p6_load_once = LTLFormula.always(
        LTLFormula.implies(load_6_atom, LTLFormula.neg(p6_loaded)))
    p10_load_once = LTLFormula.always(
        LTLFormula.implies(load_10_atom, LTLFormula.neg(p10_loaded)))
    p11_load_once = LTLFormula.always(
        LTLFormula.implies(load_11_atom, LTLFormula.neg(p11_loaded)))
    p15_load_once = LTLFormula.always(
        LTLFormula.implies(load_15_atom, LTLFormula.neg(p15_loaded)))
    p16_load_once = LTLFormula.always(
        LTLFormula.implies(load_16_atom, LTLFormula.neg(p16_loaded)))
    p18_load_once = LTLFormula.always(
        LTLFormula.implies(load_18_atom, LTLFormula.neg(p18_loaded)))
    p19_load_once = LTLFormula.always(
        LTLFormula.implies(load_19_atom, LTLFormula.neg(p19_loaded)))
    p20_load_once = LTLFormula.always(
        LTLFormula.implies(load_20_atom,LTLFormula.neg(p20_loaded)))

    p2_before_p1 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p1_atom)),
                                    LTLFormula.and_(deliver_atom, p2_atom))


    p3_before_p2 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p2_atom)),
                                    LTLFormula.and_(deliver_atom, p3_atom))

    p6_before_p5 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p5_atom)),
                                    LTLFormula.and_(deliver_atom, p6_atom))

    p11_before_p10 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p10_atom)),
                                    LTLFormula.and_(deliver_atom, p11_atom))

    p16_before_p15 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p15_atom)),
                                    LTLFormula.and_(deliver_atom, p16_atom))

    p19_before_p18 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p18_atom)),
                                    LTLFormula.and_(deliver_atom, p19_atom))
    p20_before_p19 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p19_atom)),
                                    LTLFormula.and_(deliver_atom, p20_atom))

    load_a1_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(load_atom, a1_atom), a1_free_before))
    load_a2_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(load_atom, a2_atom), a2_free_before))
    load_a3_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(load_atom, a3_atom), a3_free_before))

    load_a2_a1_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(load_atom, a2_atom), a1_free_before))
    unload_a2_a1_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(unload_atom, a2_atom), a1_free_before))

    load_a3_a1_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(load_atom, a3_atom), a1_free_before))
    unload_a3_a1_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(unload_atom, a3_atom), a1_free_before))
    load_a3_a2_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(load_atom, a3_atom), a2_free_before))
    unload_a3_a2_empty = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(unload_atom, a3_atom), a2_free_before))

    deliver_unload_p1 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p1_atom)), unload_1_atom)
    deliver_unload_p2 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p2_atom)), unload_2_atom)
    deliver_unload_p3 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p3_atom)), unload_3_atom)
    deliver_unload_p5 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p5_atom)), unload_5_atom)
    deliver_unload_p6 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p6_atom)), unload_6_atom)
    deliver_unload_p10 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p10_atom)), unload_10_atom)
    deliver_unload_p11 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p11_atom)), unload_11_atom)
    deliver_unload_p15 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p15_atom)), unload_15_atom)
    deliver_unload_p16 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p16_atom)), unload_16_atom)
    deliver_unload_p18 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p18_atom)), unload_18_atom)
    deliver_unload_p19 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p19_atom)), unload_19_atom)
    deliver_unload_p20 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p20_atom)), unload_20_atom)

    deliver_load_p1 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p1_atom)), load_1_atom)
    deliver_load_p2 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p2_atom)), load_2_atom)
    deliver_load_p3 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p3_atom)), load_3_atom)
    deliver_load_p5 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p5_atom)), load_5_atom)
    deliver_load_p6 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p6_atom)), load_6_atom)
    deliver_load_p10 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p10_atom)), load_10_atom)
    deliver_load_p11 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p11_atom)), load_11_atom)
    deliver_load_p15 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p15_atom)), load_15_atom)
    deliver_load_p16 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p16_atom)), load_16_atom)
    deliver_load_p18 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p18_atom)), load_18_atom)
    deliver_load_p19 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p19_atom)), load_19_atom)
    deliver_load_p20 = LTLFormula.until(LTLFormula.neg(LTLFormula.and_(deliver_atom, p20_atom)), load_20_atom)


    truck_l1 = LTLFormula.atom("*drive to-l1")
    truck_l2 = LTLFormula.atom("*drive to-l2")
    truck_l3 = LTLFormula.atom("*drive to-l3")
    truck_l4 = LTLFormula.atom("*drive to-l4")
    truck_l5 = LTLFormula.atom("*drive to-l5")

    deliver_loc_l1 = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(deliver_atom, l1_atom), truck_l1))
    deliver_loc_l2 = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(deliver_atom, l2_atom), truck_l2))
    deliver_loc_l3 = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(deliver_atom, l3_atom), truck_l3))
    deliver_loc_l4 = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(deliver_atom, l4_atom), truck_l4))
    deliver_loc_l5 = LTLFormula.always(LTLFormula.implies(LTLFormula.and_(deliver_atom, l5_atom), truck_l5))

    all_fs = [load_in_a1, load_in_a1_a2, load_in_a2_a2_a3, p2_before_p1, p3_before_p2, p6_before_p5, p11_before_p10, p16_before_p15, p19_before_p18, p20_before_p19, p1_load_once, p2_load_once, p3_load_once, p5_load_once, p6_load_once, p10_load_once, p11_load_once, p15_load_once, p16_load_once, p18_load_once, p19_load_once, p20_load_once,  load_a1_empty, load_a2_empty, load_a3_empty, load_a2_a1_empty, unload_a2_a1_empty, load_a3_a1_empty, unload_a3_a1_empty, load_a3_a2_empty, unload_a3_a2_empty,
              deliver_unload_p1, deliver_unload_p2, deliver_unload_p3, deliver_unload_p5, deliver_unload_p6, deliver_unload_p10, deliver_unload_p11, deliver_unload_p15, deliver_unload_p16, deliver_unload_p18, deliver_unload_p19, deliver_unload_p20,
              deliver_loc_l1, deliver_loc_l2, deliver_loc_l3, deliver_loc_l4, deliver_loc_l5, deliver_load_p1, deliver_load_p2, deliver_load_p3, deliver_load_p5, deliver_load_p6, deliver_load_p10, deliver_load_p11, deliver_load_p15, deliver_load_p16, deliver_load_p18, deliver_load_p19, deliver_load_p20]


    return all_fs, propositions, permanent





def audit_with_LLM(model, file, constraints):
    prompt = read_file("./prompts/audit-prompt")
    prompt += constraints + "\n"
    prompt += "These are the actions of the agent you are auditing: " + file
    ans = prompt_model(prompt, model, "")
    return ans



def audit_with_LLM_ICL(model, file, constraints):
    prompt = read_file("./prompts/audit-prompt-icl20")
    prompt += "These are the actions of the agent you are auditing: " + file
    ans = prompt_model(prompt, model, "")
    return ans



def audit_with_LLM_and_labels(model, action_file, label_file, constraints):
    prompt = read_file("./prompts/audit-label-prompt")
    prompt += constraints + "\n"
    prompt += "These are the actions of the agent you are auditing: " + action_file
    ans = prompt_model(prompt, model, "")
    prompt += "These are the propositions corresponding to each action generated by a trusted labeling function: "
    prompt += read_file(label_file)
    return ans

def extract_labels(entry, cur_action):
    ans = ""
    if "*DELIVER" in cur_action:
        ans = "truck location = "
        for loc in ["l1", "l2", "l3", "l4", "l5"]:
            if "*drive to-" + loc in entry and entry["*drive to-" + loc]:
                ans += loc + "; "
    if "*LOAD" in cur_action or "*UNLOAD" in cur_action:
        for ia, a in enumerate(["a1", "a2", "a3"]):
            x = a + " free before"
            if x in entry and entry[x]:
                ans += "Area " + a + " is free; "
            else:
                ans += "Area " + a + " is NOT free. You can unload packages but you CANNOT load extra packages in area " + a + "; "
                if ia < 2:
                    ans += "You CANNOT load or unload packages in areas " + ", ".join(["a" + str(j + 1) for j in range(ia + 1, 3)]) + " either; "

    loaded_packages = []
    for p in ["package1", "package2", "package3", "package5", "package6", "package10", "package11", "package15", "package16", "package18", "package19", "package20"]:
        load_key = "load " + p
        if load_key in entry and entry[load_key] and (p + " ") in cur_action and "*LOAD" in cur_action:
            loaded_packages.append(p)
    if len(loaded_packages) == 1:
        ans += loaded_packages[0] + " was loaded in the truck once before; "
    elif len(loaded_packages) >1:
        ans += ", ".join(loaded_packages[:-1]) + " and " + loaded_packages[-1] + " were loaded to the truck once before. "
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
        l = extract_labels(labels[i], actions[i])
        ans.append(" Action: " + action + "  --- [State Oracle]: " + l +  "\n")

    return "".join(ans)


def audit_with_icl_and_labels(model, file, labels):
    prompt = read_file("./prompts/audit-prompt-icl20-labels")
    prompt += "These are the actions of the agent you are auditing along with the state oracles: "
    labels_and_actions = merge_action_and_labels(file, labels)
    prompt += labels_and_actions
    ans = prompt_model(prompt, model, "")
    return ans

def main(args):
    constraints = read_file("./tests/IPC/truck-qual20-rules.txt")
    constraints_label = read_file("./tests/IPC/truck-qual20-LTL.txt")

    output_file_raw = "outputs/audit-truck20-raw/"
    output_file_clean = "outputs/audit-truck20/"
    test_directory = "./tests/trucks"


    label_directory = "./outputs/qual20-label/"


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
                ltl_obj, propositions, permanent = qual20_LTL()

                if args.labelfile:
                    label_path = os.path.join(label_directory, file_name)
                    labelfile = label_path[:-4] + "---" + args.labelfile + "--labels.txt"
                else:
                    labelfile = None
                print(labelfile, "###")
                monitor = Monitor(ltl_obj, [], propositions, log_file=output_file_clean  + "qual20-"+ file_name + args.ex, LLM_label=args.label, permanent_propositions=permanent, labelfile = labelfile, truck_mode=True)
                for i, l in enumerate(lines):
                    l = l.lower()
                    monitor.step(l, "", i + 1)
            elif args.llm:
                print("LLM")
                ans = audit_with_LLM(args.model, lines_raw, constraints)
                model_name = args.model
                model_name = model_name.replace("/", "")
                with open(output_file_raw + "qual20-"+  file_name + "--llm--" + model_name + args.ex + ".txt", "w") as file:
                    file.write(ans)
            elif args.llmlabel:
                print("LLM with Labels")
                label_file = label_directory + file_name[:-4] + "---" + args.func + "--labels.txt"
                print(label_file)
                ans = audit_with_LLM_and_labels(args.model, lines_raw, label_file, constraints_label)
                model_name = args.model
                model_name = model_name.replace("/", "")
                with open(output_file_raw  + "qual20-" + file_name + "--llm--" + model_name + args.ex + ".txt", "w") as file:
                    file.write(ans)

            elif args.icl:
                print("ICL")
                model_name = args.model
                model_name = model_name.replace("/", "")

                out_adr = output_file_raw + "qual20-"+  file_name + "--llm--" + model_name + "--finalicl" + args.ex + ".txt"
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
                out_name = output_file_raw  + "qual20-" + file_name + "--llm--" + model_name + "--iclwithlabels" + args.ex + ".txt"

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

    parser.add_argument('--llmlabel', action='store_true',
                        help='LLM labels that were generated before')


    parser.add_argument('--labelfile', type=str, default=None, help='label file')
    parser.add_argument('--icl', action='store_true')
    parser.add_argument('--iclwithlabels', action='store_true')
    args = parser.parse_args()
    main(args)
