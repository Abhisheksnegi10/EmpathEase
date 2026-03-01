"""End-to-end test for Text Emotion v2.1 integration."""
import sys
sys.path.insert(0, r'd:\EmpathEase v1\backend')

from app.ml.text_emotion import TextEmotionClassifier, get_model_hash

print("Model hash:", get_model_hash())
print()

classifier = TextEmotionClassifier()

print("=" * 70)
print("  END-TO-END TEST: 5-turn Hinglish therapy session")
print("=" * 70)

turns = [
    ("Hello aaj thoda stressed feel ho raha hai", "hinglish"),
    ("Kaam pe bahut pressure hai samajh nahi aata", "hinglish"),
    ("Sab meri galti hai main kuch theek nahi kar pata", "hinglish"),
    ("Koi nahi samajhta mujhe bilkul akela hun", "hinglish"),
    ("Haan sab theek hai koi baat nahi", "hinglish"),
]

for i, (text, lang) in enumerate(turns):
    r = classifier.analyze(text, language=lang, turn_number=i+1, context_turns=0)
    incong = ""
    if r.incongruence.get("detected"):
        incong = " | INCONGRUENCE: surface=%s implied=%s" % (
            r.incongruence["surface"], r.incongruence["implied"]
        )
    flags = r.therapeutic_flags
    print(
        "Turn %d: %-12s (%3.0f%%) | traj=%-11s | crisis=%-8s | "
        "cog=%-20s | flags=%d primary=%-20s | %3.0fms%s"
        % (
            i + 1,
            r.dominant_emotion,
            r.confidence * 100,
            r.trajectory,
            r.crisis_level,
            r.cognitive_patterns.get("dominant_pattern", "none"),
            flags.get("flag_count", 0),
            flags.get("primary_flag", "none"),
            r.inference_time_ms,
            incong,
        )
    )

# Full schema dump for last turn
print()
print("=" * 70)
print("  FULL SCHEMA (last turn):")
print("=" * 70)
schema = r.to_dict()
for k, v in schema.items():
    if isinstance(v, dict):
        print("  %s:" % k)
        for kk, vv in v.items():
            print("    %s: %s" % (kk, vv))
    else:
        print("  %s: %s" % (k, v))

print()
print("Model version:", r.model_version)
print()
print("ALL SCHEMA KEYS:", sorted(schema.keys()))
print()
print("END-TO-END TEST COMPLETE")
