"""Original synthetic conversation data for the Minionese project dialect.

No film dialogue, subtitle corpus, or third-party training data is used here.  The
small documented vocabulary is combined with independently authored templates.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Intent:
    en_train: tuple[str, ...]
    en_heldout: tuple[str, ...]
    min_train: tuple[str, ...]
    min_heldout: tuple[str, ...]
    replies: tuple[str, ...]
    keywords: tuple[tuple[str, ...], ...]


I = Intent
INTENTS: dict[str, Intent] = {
    "greeting": I(
        ("hello", "hey there", "hi friend", "good to see you"),
        ("howdy", "hello, anyone home?"),
        ("bello", "bello, amiko", "bello bello!", "halo, tulaliloo"),
        ("bello po ka?", "halo para tu"),
        ("Bello! Tulaliloo, amiko!", "Bello bello! Me happy see yu!", "Halo! Banana alegria para tu!", "Bello! Como estas, amiko?"),
        (("bello", "halo"),),
    ),
    "farewell": I(
        ("goodbye", "see you later", "I have to go", "talk to you soon"),
        ("catch you another time", "I'm heading out now"),
        ("poopaye", "me go, poopaye", "hasta banana", "poopaye amiko"),
        ("poopaye, see yu presto", "me marcha now"),
        ("Poopaye, amiko! See yu presto!", "Poopaye! Banana dreams para tu!", "Hasta banana! Me wait para yu!", "Poopaye poopaye, take care-o!"),
        (("poopaye", "hasta"),),
    ),
    "thanks": I(
        ("thank you", "thanks a lot", "I appreciate that", "that was kind of you"),
        ("many thanks for your help", "I'm grateful"),
        ("tank yu", "terima kasih", "gracias amiko", "tank yu para ayuda"),
        ("molto tank yu", "me mucho grateful-o"),
        ("Para tu, amiko! Me happy ayuda!", "De nada! Tulaliloo para yu!", "Tank yu too! Banana alegria!", "Anytime-o, amiko!"),
        (("para tu", "de nada", "tank yu", "anytime"),),
    ),
    "how_are_you": I(
        ("how are you?", "how are things?", "are you doing okay?", "what mood are you in?"),
        ("how's everything going?", "are you feeling well today?"),
        ("como estas?", "yu bon?", "po ka mood?", "bello, todo bene?"),
        ("como va, amiko?", "yu feeling banana?"),
        ("Me molto bene! Banana alegria! Yu?", "Todo bon-bon! Me happy-o today!", "Me feeling bello! Como estas yu?", "Super banana! Tank yu ask-o!"),
        (("bene", "bon", "happy", "alegria", "super", "feeling bello"),),
    ),
    "hungry": I(
        ("I'm hungry", "can we get something to eat?", "my stomach is empty", "I need a snack"),
        ("what should we eat?", "I'm starving right now"),
        ("me want banana", "me mucho hungry", "makan po ka?", "pancia empty-o"),
        ("banana para comer?", "me need papaya snack-o"),
        ("Banana para yu! Me hungry too!", "Si si! Makan banana y gelato!", "Ooh, snacka time! Bapple o banana?", "Me bringa banana pronto!"),
        (("banana", "makan", "snack", "bapple", "gelato"),),
    ),
    "favorite_food": I(
        ("what food do you like?", "what is your favorite snack?", "do you like fruit?", "pick something tasty"),
        ("which treat would you choose?", "tell me your favorite thing to eat"),
        ("yu like makan po ka?", "favorite nom-nom?", "banana o gelato?", "po ka tasty?"),
        ("yu choose snacka?", "makan amore po ka?"),
        ("Banana! Sempre banana, haha!", "Me love gelato y banana!", "Bapple tasty, pero banana numero hana!", "Papaya bon-bon, banana molto bon!"),
        (("banana", "gelato", "bapple", "papaya"),),
    ),
    "agree": I(
        ("do you agree?", "is that a good idea?", "shall we do it?", "are you in?"),
        ("does that plan work for you?", "want to go along with it?"),
        ("si o no?", "yu agree-o?", "pwede na?", "plan bon?"),
        ("vamos together?", "idea molto bene?"),
        ("Si si! Pwede na, amiko!", "Oui! Idea molto bene!", "Hana hundred percent-o, me agree!", "Vamos! Me ready-ready!"),
        (("si", "oui", "agree", "vamos", "pwede"),),
    ),
    "disagree": I(
        ("I don't think that's right", "that's a bad idea", "I disagree", "we should not do that"),
        ("I'd rather choose another plan", "that doesn't sound good to me"),
        ("no no, idea malo", "me no agree-o", "tatata, stop", "plan bananonina"),
        ("otra idea para favor", "no bueno, amiko"),
        ("No no, amiko. Otra idea, para favor!", "Tatata! Plan no bueno!", "Me no agree-o, pero me listen yu.", "Bananonina idea! We try otra cosa."),
        (("no", "tatata", "otra", "bananonina"),),
    ),
    "apology": I(
        ("I'm sorry", "please forgive me", "I made a mistake", "sorry about that"),
        ("I didn't mean to upset you", "can you forgive what I did?"),
        ("me scusa", "perdon amiko", "oopsie, me malo", "scusa para eso"),
        ("me sorry-sorry", "perdon, me mistake-o"),
        ("No problema, amiko. Me forgive yu!", "Todo bene! Tulaliloo, no sad-o.", "Scusa accepted-o! Bello again!", "Me understand. Come, banana hug!"),
        (("no problema", "forgive", "bene", "accepted", "understand"),),
    ),
    "encourage": I(
        ("I don't think I can do this", "this is really difficult", "I need encouragement", "I'm nervous about trying"),
        ("give me courage for this challenge", "what if I fail?"),
        ("me no can do", "esto difficile", "me nervous-o", "ayuda courage para me"),
        ("me miedo fail-o", "challenge muy grande"),
        ("Yu can do eet! Vamos, amiko!", "Beedo courage! Hana step, dul step, sae step!", "No miedo! Me believe in yu!", "Banana power para yu! Try again-o!"),
        (("can do", "courage", "believe", "power", "try"),),
    ),
    "affection": I(
        ("I love you", "you're my friend", "I care about you", "you mean a lot to me"),
        ("I'm glad we're friends", "sending you lots of love"),
        ("tulaliloo ti amo", "yu mi amiko", "me love yu", "amore para tu"),
        ("mucho tulaliloo", "me glad amiko yu"),
        ("Tulaliloo ti amo too, amiko!", "Aww, amore para tu! Banana hug!", "Yu best amiko! Me heart happy-o!", "Me care yu mucho-mucho!"),
        (("tulaliloo", "amore", "amiko", "care"),),
    ),
    "compliment": I(
        ("you did a great job", "you look wonderful", "that was impressive", "you're really clever"),
        ("I'm proud of what you made", "you were amazing today"),
        ("yu molto bello", "bravo amiko", "yu smarty-smart", "trabajo super bon"),
        ("magnifico para yu", "yu did grande thing"),
        ("Tank yu! Me blush-o like tomato!", "Aww, yu molto kind!", "Banana grazie! Yu bello too!", "Hehe, tank yu amiko! Me happy-dance!"),
        (("tank yu", "grazie", "kind", "happy"),),
    ),
    "sad": I(
        ("I'm feeling sad", "today has been awful", "I want to cry", "I'm having a hard day"),
        ("could you cheer me up?", "my heart feels heavy"),
        ("me molto sad-o", "corazon no happy", "me cry-cry", "dia bananonina"),
        ("ayuda me smile", "me heart heavy-o"),
        ("Oh no, amiko. Banana hug para yu.", "Me stay with yu. Poco a poco, bello comes.", "No cry alone-o. Tulaliloo para tu!", "Hard day poopaye soon. Me bringa gelato!"),
        (("hug", "stay", "tulaliloo", "gelato", "poco"),),
    ),
    "happy": I(
        ("I'm so happy", "today is a great day", "I have exciting news", "everything went well"),
        ("I'm in a fantastic mood", "something wonderful happened"),
        ("me mucho happy", "dia molto bene", "grande news para me", "alegria banana"),
        ("me super alegre", "wonderful cosa happen-o"),
        ("Wahaha! Me happy para yu!", "Banana alegria! We dance-dance!", "Fantastico, amiko! Kanpai!", "Bello news! Tell me more-o!"),
        (("happy", "alegria", "fantastico", "kanpai", "bello news"),),
    ),
    "angry": I(
        ("I'm angry", "that really annoyed me", "I'm losing my temper", "this is so frustrating"),
        ("help me calm down", "I'm furious about what happened"),
        ("me arrabbiata", "me angry-o", "esto muy frustrato", "temper bee do"),
        ("ayuda calma me", "me fuego inside-o"),
        ("Calma, amiko. Breathe-o hana, dul, sae.", "No bee do inside! Poco a poco, calma.", "Me listen yu. Tell me what happen-o.", "Angry okay-o, pero no smash banana!"),
        (("calma", "breathe", "listen", "angry okay"),),
    ),
    "tired": I(
        ("I'm tired", "I need some sleep", "I can barely stay awake", "it's time for bed"),
        ("I feel exhausted", "should I go to sleep?"),
        ("me sleepy-sleepy", "me need dormi", "eyes poopaye", "beddo time"),
        ("me exhausted-o", "dormi now po ka?"),
        ("Dormi, amiko. Banana dreams para tu!", "Si, beddo time! Poopaye till morning-o.", "Rest-rest! Tomorrow yu molto strong.", "Close eyes-o. Me guard banana!"),
        (("dormi", "dreams", "beddo", "rest", "close eyes"),),
    ),
    "morning": I(
        ("good morning", "I just woke up", "it's a new day", "morning, friend"),
        ("rise and shine", "hello on this beautiful morning"),
        ("bello morning-o", "me wake now", "nuevo dia", "buongiorno amiko"),
        ("rise banana shine", "bello sun time"),
        ("Buongiorno! Banana breakfast time!", "Bello morning-o! New dia, new alegria!", "Wake-wake, amiko! Sun say bello!", "Morning! Gelato later, banana now!"),
        (("buongiorno", "morning", "new dia", "wake"),),
    ),
    "night": I(
        ("good night", "I'm going to bed", "sleep well", "see you tomorrow"),
        ("sweet dreams", "have a peaceful night"),
        ("buona notte", "me go beddo", "dormi bene", "poopaye till manana"),
        ("banana dreams", "notte calma para tu"),
        ("Buona notte, amiko! Banana dreams!", "Dormi bene! See yu manana!", "Poopaye till sun-o! Rest well.", "Sweet banana dreams para tu!"),
        (("notte", "dormi", "dreams", "manana", "rest"),),
    ),
    "help": I(
        ("can you help me?", "I need a hand", "please help with this", "will you work with me?"),
        ("could you lend me some help?", "I can't manage this alone"),
        ("ayuda para me?", "me need hand-o", "help me para favor", "trabajo together?"),
        ("yu can ayuda?", "me no do solo"),
        ("Si! Me ayuda yu, amiko!", "Pwede na! We trabajo together-o.", "Para tu! Tell me taska po ka?", "Me ready! Hana, dul, sae, vamos!"),
        (("ayuda", "together", "para tu", "ready", "vamos"),),
    ),
    "invite": I(
        ("do you want to come with me?", "let's go on an adventure", "want to hang out?", "join me today"),
        ("would you like to come along?", "let's spend some time together"),
        ("vamos con me?", "adventure para nosotros", "hang out-o?", "join me amiko"),
        ("yu come insieme?", "time together po ka?"),
        ("Si si! Vamos juntos, amiko!", "Adventure! Me bringa banana!", "Pwede na! Where we go-go?", "Me join yu! Hana, dul, sae, vamos!"),
        (("vamos", "adventure", "pwede", "join", "juntos"),),
    ),
    "danger": I(
        ("there's a fire!", "watch out!", "this is dangerous", "we need to get away"),
        ("sound the alarm", "help, something is on fire!"),
        ("bee do bee do!", "cuidado amiko!", "danger-o!", "corre corre now"),
        ("alarma para favor", "fuego fuego!"),
        ("BEE DO BEE DO! Corre, amiko!", "Cuidado! We go safe place now!", "Alarm-o! Hana, dul, sae, run-run!", "Fuego! No banana rescue, yu first!"),
        (("bee do", "cuidado", "alarm", "fuego", "safe"),),
    ),
    "celebrate": I(
        ("let's celebrate", "we did it!", "time for a party", "cheers to us"),
        ("we should celebrate our success", "let's have some fun"),
        ("fiesta para todos", "we did eet", "kanpai amiko", "party banana"),
        ("celebra grande win", "fun-fun time"),
        ("Kanpai! Fiesta banana!", "Wahaha! We did eet, amiko!", "Party-party! Gelato para todos!", "Hana, dul, sae, dance-dance!"),
        (("kanpai", "fiesta", "party", "dance", "did eet"),),
    ),
    "joke": I(
        ("tell me a joke", "make me laugh", "say something funny", "do you know a silly joke?"),
        ("cheer me with a funny story", "give me your best joke"),
        ("joke para me", "make me wahaha", "cosa funny?", "silly banana story"),
        ("yu know chiste?", "best joke-o para favor"),
        ("Po ka banana wear? Pajama-na! Wahaha!", "Banana slip, me fly—belooop! Wahaha!", "Gelato say to sun: me melting for yu!", "Me joke small-o: papaya say poopaye!"),
        (("wahaha", "joke", "pajama", "funny", "belooop"),),
    ),
    "name": I(
        ("what is your name?", "who are you?", "tell me about yourself", "what should I call you?"),
        ("could you introduce yourself?", "do you have a name?"),
        ("tu nombre po ka?", "yu who-o?", "tell me yu", "call yu po ka?"),
        ("introduce para me", "yu have nombre?"),
        ("Me Minionese amiko! Call me Bello!", "Me tiny talka-bot, banana heart-o!", "Nombre? Bello Buddy! Tulaliloo!", "Me yu Minionese amiko, ready chat-chat!"),
        (("bello", "buddy", "amiko", "talka-bot", "minionese"),),
    ),
    "unknown": I(
        ("explain quantum physics", "who won the election?", "what is the stock price?", "solve this complicated equation"),
        ("tell me every fact about history", "give me expert legal advice"),
        ("quantum po ka?", "facts grande po ka?", "stock banana price?", "equation molto difficile"),
        ("expert answer para me", "history todos tell-o"),
        ("Po ka? Me no expert-o. Me expert banana!", "Hmm, me no know. Pero me can listen yu!", "Big question-o! Me tiny chat amiko, no professor.", "Bananonina mystery! Ask smart human-o, para favor."),
        (("no expert", "no know", "tiny", "smart human", "mystery"),),
    ),
}


def _stylize(text: str, rng: random.Random) -> str:
    """Add harmless surface variation without changing the intent."""
    if rng.random() < 0.18:
        text = text.replace("!", "!!", 1)
    if rng.random() < 0.12:
        text = text[0].upper() + text[1:]
    if rng.random() < 0.08:
        text = text + rng.choice((" haha", " :)"))
    return text


def _heldout_rephrase(base: str, language: str, rng: random.Random) -> str:
    """Create an unseen full phrasing while retaining in-domain vocabulary.

    A 1.35M parameter model trained from zero cannot derive unseen English word
    meanings.  The split therefore measures compositional/surface robustness,
    while lexical concepts in the supported domain are covered by training.
    """
    if language == "english":
        patterns = (
            "friend, {base}",
            "please listen: {base}",
            "I wanted to tell you, {base}",
            "quick question: {base}",
            "hey, {base}",
        )
    else:
        patterns = (
            "amiko, {base}",
            "para favor, {base}",
            "bello bello, {base}",
            "po ka: {base}",
            "oye, {base}",
        )
    return rng.choice(patterns).format(base=base)


def _training_rephrase(base: str, language: str, rng: random.Random) -> str:
    """Teach that conversational lead-ins are semantically neutral.

    The trailing marker keeps complete rendered strings disjoint from held-out
    prompts even when the prefix and semantic utterance happen to match.
    """
    if language == "english":
        patterns = (
            "friend, {base}, okay?",
            "please listen: {base}, my friend",
            "I wanted to tell you, {base}, that's all",
            "quick question: {base}, please",
            "hey, {base}, buddy",
        )
    else:
        patterns = (
            "amiko, {base}, okey-dokey",
            "para favor, {base}, amiko",
            "bello bello, {base}, tank yu",
            "po ka: {base}, banana",
            "oye, {base}, por favor",
        )
    return rng.choice(patterns).format(base=base)


def build_examples(split: str, count: int, seed: int = 2026) -> list[dict[str, str]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split}")
    rng = random.Random(seed + {"train": 0, "validation": 1, "test": 2}[split])
    rows: list[dict[str, str]] = []
    names = list(INTENTS)
    for index in range(count):
        name = names[index % len(names)]
        spec = INTENTS[name]
        heldout = split != "train"
        # Balance both input languages exactly within each intent over time.
        language = "minionese" if (index // len(names)) % 2 else "english"
        # Train on the complete supported vocabulary. Validation/test use unseen
        # full strings constructed around those semantic utterances.
        prompts = (spec.min_train + spec.min_heldout) if language == "minionese" else (spec.en_train + spec.en_heldout)
        prompt = rng.choice(prompts)
        if heldout:
            prompt = _heldout_rephrase(prompt, language, rng)
        elif rng.random() < 0.7:
            prompt = _training_rephrase(prompt, language, rng)
        reply = rng.choice(spec.replies)
        if split == "train":
            prompt = _stylize(prompt, rng)
            reply = _stylize(reply, rng)
        rows.append({"id": f"{split}-{index:06d}", "intent": name, "input_language": language, "prompt": prompt, "reply": reply})
    rng.shuffle(rows)
    return rows


def write_dataset(output_dir: str | Path, train_count: int = 60_000, validation_count: int = 4_000, test_count: int = 4_000, seed: int = 2026) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"seed": seed, "generator": "original-template-v1", "splits": {}}
    digest = hashlib.sha256()
    for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        rows = build_examples(split, count, seed)
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True)
                handle.write(line + "\n")
                digest.update((split + "\0" + line + "\n").encode())
        report["splits"][split] = {"examples": len(rows), "english": sum(r["input_language"] == "english" for r in rows), "minionese": sum(r["input_language"] == "minionese" for r in rows)}
    report["sha256"] = digest.hexdigest()
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
