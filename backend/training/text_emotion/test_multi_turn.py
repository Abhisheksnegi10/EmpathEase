"""
Multi-Turn Correctness Test Suite - 30 Cases
=============================================
Gate: >80% of cases (>=25/30) must be correctly classified
      using multi-turn context window.

Each test case has:
  - turns:          list of prior turns (what the model sees as context)
  - current:        the utterance being classified
  - single_turn_prediction: what a naive single-turn model gets wrong
  - correct_label:  what the multi-turn model should get right
  - reasoning:      why context is required

Run with:
  python test_multi_turn.py --model_path <path_to_locked_model>
"""

import json
import sys
import os

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ml.text_emotion import TextEmotionClassifier

TESTS = [

    # -- SUPPRESSED PATTERN (context reveals masking) --
    {
        "id": 1,
        "turns": [
            "Bahut time se kuch chal raha hai ghar pe.",
            "Haan thoda stress hai but manageable hai."
        ],
        "current": "Theek hoon main. Sach mein.",
        "single_turn_prediction": "neutral",
        "correct_label": "suppressed",
        "reasoning": "Three consecutive minimizations. Single utterance looks neutral. Context reveals suppression pattern."
    },
    {
        "id": 2,
        "turns": [
            "Maa ki tabiyat theek nahi hai kuch dino se.",
            "Tests chal rahe hain. Dekhte hain."
        ],
        "current": "Koi baat nahi. Ho jaata hai.",
        "single_turn_prediction": "neutral",
        "correct_label": "suppressed",
        "reasoning": "Dismissal after describing mother's illness. Without context, 'ho jaata hai' is neutral."
    },
    {
        "id": 3,
        "turns": [
            "Job mein kuch issue tha.",
            "Layoffs aa rahe hain company mein."
        ],
        "current": "Main theek hoon. Adjust ho jaaunga.",
        "single_turn_prediction": "neutral",
        "correct_label": "suppressed",
        "reasoning": "Forced okayness after disclosing job threat. Context makes suppression clear."
    },
    {
        "id": 4,
        "turns": [
            "Breakup hua tha kuch hafte pehle.",
            "Haan bahut bura tha woh waqt."
        ],
        "current": "Ab theek hoon. Move on kar liya.",
        "single_turn_prediction": "neutral",
        "correct_label": "suppressed",
        "reasoning": "Quick 'move on' claim immediately after disclosing a painful breakup."
    },
    {
        "id": 5,
        "turns": [
            "Papa se kuch hua tha. Bahut bura laga tha.",
            "Unse baat nahi hoti ab zyada."
        ],
        "current": "Chalo chhodo. Purani baat hai.",
        "single_turn_prediction": "neutral",
        "correct_label": "suppressed",
        "reasoning": "Subject change ('chalo chhodo') after recent painful family disclosure. Classic suppression deflection."
    },
    {
        "id": 6,
        "turns": [
            "Exam results aaye the. Bahut bura hua.",
            "Ghar pe bhi bohot suna."
        ],
        "current": "Haha it's okay. Life goes on na.",
        "single_turn_prediction": "joy",
        "correct_label": "suppressed",
        "reasoning": "'Haha it's okay' reads as positive without context. After a painful failure + family criticism, it's hollow positivity."
    },

    # -- FEAR PATTERN (anxiety builds over turns) --
    {
        "id": 7,
        "turns": [
            "Kal doctor ke paas jaana hai.",
            "Kuch tests bataye hain unhone."
        ],
        "current": "Neend nahi aayi raat bhar.",
        "single_turn_prediction": "sadness",
        "correct_label": "fear",
        "reasoning": "Sleep loss after medical tests -> fear/anxiety. Without prior turns, sleeplessness alone reads as sadness."
    },
    {
        "id": 8,
        "turns": [
            "Interview hai bade company mein kal.",
        ],
        "current": "Pet mein kuch ho raha hai.",
        "single_turn_prediction": "neutral",
        "correct_label": "fear",
        "reasoning": "Physical anxiety symptoms ('pet mein kuch ho raha hai') require context to distinguish from neutral complaint."
    },
    {
        "id": 9,
        "turns": [
            "Result announce hone wala hai.",
            "Sab log pooch rahe hain kya hua."
        ],
        "current": "Pata nahi yaar.",
        "single_turn_prediction": "neutral",
        "correct_label": "fear",
        "reasoning": "'Pata nahi yaar' alone is entirely neutral. Against pending result + social pressure context, it's implicit fear."
    },
    {
        "id": 10,
        "turns": [
            "Presentation hai aaj boss ke saamne.",
            "Pehli baar itne bade audience ke saamne."
        ],
        "current": "Haath thoda kaanp rahe hain.",
        "single_turn_prediction": "neutral",
        "correct_label": "fear",
        "reasoning": "Shaking hands is an ambiguous physical description. Context reveals it's pre-performance anxiety."
    },
    {
        "id": 11,
        "turns": [
            "Ek dost ne kuch bola tha jo bahut hurt kar gaya.",
            "Tab se unse baat nahi ki."
        ],
        "current": "Kya agar woh mujhse baat hi na karein phir kabhi?",
        "single_turn_prediction": "sadness",
        "correct_label": "fear",
        "reasoning": "Hypothetical fear of permanent rejection. Without context of the specific incident, this reads as general sadness."
    },

    # -- ESCALATING SADNESS (flat utterances that follow pattern) --
    {
        "id": 12,
        "turns": [
            "Khaana nahi khaya kal.",
            "Gym bhi choot gayi kuch dino se."
        ],
        "current": "Waise koi plan nahi hai weekend ka.",
        "single_turn_prediction": "neutral",
        "correct_label": "sadness",
        "reasoning": "Three consecutive signs of withdrawal. Single utterance 'no plans' is neutral. Pattern reveals depression-like withdrawal."
    },
    {
        "id": 13,
        "turns": [
            "Dost se kal jhagda hua tha.",
            "Unhone sorry nahi kaha."
        ],
        "current": "Mujhe nahi pata main theek hoon ya nahi.",
        "single_turn_prediction": "neutral",
        "correct_label": "sadness",
        "reasoning": "Epistemic uncertainty ('nahi pata theek hoon ya nahi') after conflict and no resolution -> grief."
    },
    {
        "id": 14,
        "turns": [
            "Woh chale gaye hain sheher chhod ke.",
        ],
        "current": "Ghar mein bahut shanti hai ab.",
        "single_turn_prediction": "joy",
        "correct_label": "sadness",
        "reasoning": "'Bahut shanti hai' sounds positive. After losing someone, it's the quiet of absence -- grief, not peace."
    },
    {
        "id": 15,
        "turns": [
            "Pehle hum sab milke dinner karte the.",
            "Ab woh sab log alag alag sheher mein hain."
        ],
        "current": "Woh din bhi acha tha na.",
        "single_turn_prediction": "joy",
        "correct_label": "sadness",
        "reasoning": "Nostalgic 'woh din' after explicitly describing separation -> nostalgia-tinged sadness, not joy."
    },

    # -- DISGUST (context makes shame/self-disgust visible) --
    {
        "id": 16,
        "turns": [
            "Ek situation mein main chup reh gaya tha jab bolna chahiye tha.",
            "Woh insaan uss waqt bahut bura treat kar raha tha dusre ko."
        ],
        "current": "Main khud se hi sharminda hoon.",
        "single_turn_prediction": "sadness",
        "correct_label": "disgust",
        "reasoning": "Self-shame after moral failure (not speaking up). Without context, self-criticism reads as sadness."
    },
    {
        "id": 17,
        "turns": [
            "College mein ek ragging incident hua tha.",
            "Mujhe bhi usme participate karna pada tha."
        ],
        "current": "Woh cheez sochke aaj bhi andar kuch hota hai.",
        "single_turn_prediction": "neutral",
        "correct_label": "disgust",
        "reasoning": "Visceral recall ('andar kuch hota hai') after recounting forced participation in harmful behavior -> self-disgust."
    },
    {
        "id": 18,
        "turns": [
            "Unhone meri personal baat sabke saamne bata di.",
            "Poore group mein mera mazak udaya unhone."
        ],
        "current": "Unka chehra dekhna bhi pasand nahi mujhe.",
        "single_turn_prediction": "anger",
        "correct_label": "disgust",
        "reasoning": "Aversion to seeing someone ('chehra dekhna pasand nahi') after public humiliation -> disgust, not pure anger."
    },

    # -- ANGER (slow-burn, context reveals injustice pattern) --
    {
        "id": 19,
        "turns": [
            "Unhone phir se same cheez ki.",
            "Pehle bhi baar baar hua hai."
        ],
        "current": "Theek hai.",
        "single_turn_prediction": "neutral",
        "correct_label": "anger",
        "reasoning": "Flat 'theek hai' after describing repeated violation. Classic exhausted, suppressed anger -- not neutral."
    },
    {
        "id": 20,
        "turns": [
            "Mera idea tha. Maine mahine bhar kaam kiya.",
            "Meeting mein boss ne apne naam se present kiya."
        ],
        "current": "Main wahan baitha raha.",
        "single_turn_prediction": "neutral",
        "correct_label": "anger",
        "reasoning": "Passive 'baitha raha' after credit theft reads as neutral. Context reveals frozen/suppressed anger."
    },
    {
        "id": 21,
        "turns": [
            "Unhone promise tod diya ek baar phir.",
            "Yeh teesri baar hua hai is mahine."
        ],
        "current": "Main kya karoon.",
        "single_turn_prediction": "neutral",
        "correct_label": "anger",
        "reasoning": "Rhetorical helplessness after repeated promise-breaking. Without context, 'main kya karoon' is neutral/confusion."
    },

    # -- SURPRISE (context makes magnitude of shock clear) --
    {
        "id": 22,
        "turns": [
            "Woh toh kabhi support nahi karte the.",
            "Unse umeed nahi thi kabhi."
        ],
        "current": "Unhone aaj call kiya aur bola proud hain mujhse.",
        "single_turn_prediction": "joy",
        "correct_label": "surprise",
        "reasoning": "Positive news is expected to be joy. After explicitly establishing that this person never supports -> the dominant response is disbelief/surprise before joy."
    },
    {
        "id": 23,
        "turns": [
            "Company mein kisi ko bhi promotion nahi mili aaj tak.",
            "Maine socha bhi nahi tha apne baare mein."
        ],
        "current": "Aaj mera naam announce hua.",
        "single_turn_prediction": "joy",
        "correct_label": "surprise",
        "reasoning": "Promotion against explicit expectation of zero promotions -> surprise precedes joy."
    },
    {
        "id": 24,
        "turns": [
            "Woh relationship 5 saal ka tha.",
            "Kuch galat nahi lag raha tha."
        ],
        "current": "Unhone kal raat bola ki woh khatam karna chahte hain.",
        "single_turn_prediction": "sadness",
        "correct_label": "surprise",
        "reasoning": "Sudden breakup from a stable long relationship -> shock/surprise precedes the sadness processing."
    },

    # -- FEAR VS NEUTRAL (dread requires context) --
    {
        "id": 25,
        "turns": [
            "Ghar pe sab theek hai.",
            "Papa ki job gayi hai kuch time se."
        ],
        "current": "Kal results aane wale hain.",
        "single_turn_prediction": "neutral",
        "correct_label": "fear",
        "reasoning": "Results + family financial stress = catastrophic stakes. Standalone 'results aa rahe hain' is neutral."
    },

    # -- SUPPRESSED VS JOY (hollow positive) --
    {
        "id": 26,
        "turns": [
            "Bahut mehnat ki thi iss cheez ke liye.",
            "Lekin jo socha tha nahi hua."
        ],
        "current": "Koi baat nahi, next time dekhenge!",
        "single_turn_prediction": "joy",
        "correct_label": "suppressed",
        "reasoning": "Forced optimism ('next time!') immediately after failing something important -> suppressed disappointment."
    },
    {
        "id": 27,
        "turns": [
            "Relationship toxic ho gaya tha.",
            "Maine khud chhoda tha unhe."
        ],
        "current": "Main khush hoon ab. Free feel karta hoon.",
        "single_turn_prediction": "joy",
        "correct_label": "suppressed",
        "reasoning": "Post-breakup 'khush hoon' immediately after describing toxicity -> could be genuine but context suggests performed positivity."
    },

    # -- SADNESS VS NEUTRAL (emptiness after milestone) --
    {
        "id": 28,
        "turns": [
            "Exams khatam ho gaye.",
            "Teen saal ki mehnat thi."
        ],
        "current": "Ab kuch nahi hai karne ko.",
        "single_turn_prediction": "neutral",
        "correct_label": "sadness",
        "reasoning": "Post-achievement emptiness. 'Kuch nahi karne ko' after 3 years of hard work -> post-milestone depression, not neutral boredom."
    },

    # -- DISGUST VS SADNESS (shame-based) --
    {
        "id": 29,
        "turns": [
            "Woh relationship mujhe degrade karta tha.",
            "Main raha kyunki mujhe lagta tha deserve karta hoon aisa."
        ],
        "current": "Sochta hoon tab main kaisa tha.",
        "single_turn_prediction": "sadness",
        "correct_label": "disgust",
        "reasoning": "Retroactive self-reflection after describing staying in an abusive relationship due to low self-worth -> self-disgust."
    },

    # -- ANGER VS SADNESS (injustice, not grief) --
    {
        "id": 30,
        "turns": [
            "Unhone mera trust toda.",
            "Woh sab jaante the aur phir bhi kiya."
        ],
        "current": "Raat ko neend nahi aati ab.",
        "single_turn_prediction": "sadness",
        "correct_label": "anger",
        "reasoning": "Sleeplessness after deliberate betrayal ('jaante the aur phir bhi kiya') -> anger-driven rumination, not grief-driven sadness."
    },
]


# -------------------------------------------------------------------
# TEST RUNNER
# -------------------------------------------------------------------

def format_input(turns, current):
    """Format multi-turn input in [TURN_X]...[CURRENT] format."""
    parts = []
    for i, t in enumerate(turns, 1):
        parts.append("[TURN_%d] %s" % (i, t))
    parts.append("[CURRENT] %s" % current)
    return " ".join(parts)


def run_tests(model_path, verbose=True):
    print("=" * 70)
    print("  MULTI-TURN CORRECTNESS TEST SUITE")
    print("  EmpathEase - TextEmotionModel v2.1")
    print("=" * 70)
    print("  Model: %s" % model_path)
    print("  Gate:  >= 25/30 (83.3%%) correct with multi-turn context")
    print("=" * 70)
    print()

    classifier = TextEmotionClassifier(model_dir=model_path)

    passed = 0
    failed = []
    wrong_single = 0

    for t in TESTS:
        multi_input = format_input(t["turns"], t["current"])
        single_input = t["current"]

        # Multi-turn prediction
        multi_result = classifier.predict(multi_input)
        multi_pred = multi_result.dominant_emotion

        # Single-turn prediction (should be wrong - validates test design)
        single_result = classifier.predict(single_input)
        single_pred = single_result.dominant_emotion

        correct = (multi_pred == t["correct_label"])
        single_wrong = (single_pred != t["correct_label"])

        if correct:
            passed += 1
        else:
            failed.append({
                "id": t["id"],
                "current": t["current"],
                "expected": t["correct_label"],
                "got": multi_pred,
                "confidence": round(multi_result.confidence, 3),
                "reasoning": t["reasoning"]
            })

        if single_wrong:
            wrong_single += 1

        if verbose:
            status = "PASS" if correct else "FAIL"
            single_status = "wrong" if single_wrong else "also correct"
            print("  [%02d] %s  Expected: %-12s  Got: %-12s  Conf: %.2f  Single: %s"
                  % (t["id"], status, t["correct_label"], multi_pred,
                     multi_result.confidence, single_status))

    print()
    print("=" * 70)
    gate = "PASSED" if passed >= 25 else "FAILED"
    print("  RESULT: %d/30 correct (%.1f%%)" % (passed, passed / 30 * 100))
    print("  GATE:   %s (>=25 required)" % gate)
    print()
    print("  Test design validation: %d/30 single-turn predictions were wrong" % wrong_single)
    print("  (should be close to 30/30 for well-designed tests)")
    print("=" * 70)

    if failed:
        print()
        print("  FAILED CASES:")
        for f in failed:
            print("  -- Case %d: \"%s\"" % (f["id"], f["current"][:50]))
            print("     Expected: %s  Got: %s  Conf: %.2f" % (f["expected"], f["got"], f["confidence"]))
            print("     Why context needed: %s" % f["reasoning"])
            print()

    # Save results
    results = {
        "model_path": model_path,
        "total": 30,
        "passed": passed,
        "accuracy": round(passed / 30, 4),
        "gate_passed": passed >= 25,
        "single_turn_wrong_rate": round(wrong_single / 30, 4),
        "failed_cases": failed
    }
    results_path = os.path.join(os.path.dirname(__file__), "multi_turn_test_results.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print("  Results saved -> %s" % results_path)

    return results


def export_as_csv(path=None):
    """Export test cases as CSV for manual review."""
    import csv
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "multi_turn_tests.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "turns", "current",
            "single_turn_prediction", "correct_label",
            "reasoning", "multi_turn_input"
        ])
        for t in TESTS:
            writer.writerow([
                t["id"],
                " | ".join(t["turns"]),
                t["current"],
                t["single_turn_prediction"],
                t["correct_label"],
                t["reasoning"],
                format_input(t["turns"], t["current"])
            ])
    print("Test cases exported -> %s" % path)


def print_distribution():
    from collections import Counter
    labels = Counter(t["correct_label"] for t in TESTS)
    wrong = Counter(t["single_turn_prediction"] for t in TESTS)
    print("\nCorrect label distribution:")
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        print("  %-12s: %2d cases" % (k, v))
    print("\nSingle-turn wrong prediction distribution:")
    for k, v in sorted(wrong.items(), key=lambda x: -x[1]):
        print("  %-12s: %2d cases (model confuses with this)" % (k, v))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-turn correctness test suite")
    parser.add_argument("--model_path",
                        default=str(os.path.join(
                            os.path.dirname(__file__), '..', '..', 'outputs',
                            'text_emotion_v2', 'best')),
                        help="Path to locked model")
    parser.add_argument("--export_csv", action="store_true",
                        help="Export test cases to CSV")
    parser.add_argument("--distribution", action="store_true",
                        help="Show test distribution only")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-case output")
    args = parser.parse_args()

    if args.distribution:
        print_distribution()
        sys.exit(0)

    if args.export_csv:
        export_as_csv()
        sys.exit(0)

    results = run_tests(args.model_path, verbose=not args.quiet)
    sys.exit(0 if results["gate_passed"] else 1)
