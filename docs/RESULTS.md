# Training and evaluation results

## Final artifact

- Random initialization; no pretrained model or tokenizer
- 1,354,560 trainable parameters
- 4 decoder layers, 160 model width, 5 heads, 640 MLP width
- 512-token locally trained byte-level BPE; 256-token context
- 5,000 batches; batch size 64
- 9,981,697 sampled non-padding training tokens, including 4,982,270 scored assistant tokens
- Best checkpoint at step 3,000
- Training time: 242 seconds on an Apple M4 with 16 GB unified memory

## Untouched test split

The 4,000-row test split uses complete prompt strings absent from training. Its supported semantic vocabulary is intentionally in-domain: a 1.35M-parameter model trained from zero cannot infer definitions for English words it never sees.

| Measure | Result |
| --- | ---: |
| Test cross-entropy | 0.1099 |
| Test perplexity | 1.1161 |
| Unigram baseline loss | 4.6882 |
| Behavioral samples | 400 |
| Intent relevance | 99.0% |
| Minionese style | 100.0% |
| Clean completion | 100.0% |
| Combined pass rate | 99.0% |
| English-input pass rate | 99.5% |
| Minionese-input pass rate | 98.5% |

Every intent scored 100% in the fixed evaluation sample except jokes, which scored 75%. Exact per-intent results and all generated outputs are in `artifacts/evaluation.json`.

## Example unseen conversations

```text
You: Hey buddy, how are you feeling today?
Minion: Me molto bene! Banana alegria! Yu?

You: Could you help me with this task?
Minion: Pwede na! We trabajo together-o.

You: What snack do you like best?
Minion: Papaya bon-bon, banana molto bon!

You: There's a fire, get to safety!
Minion: Fuego! No banana rescue, yu first!

You: Tell me your best silly joke.
Minion: Banana slip, me fly—belooop! Wahaha!

You: bello amiko, como estas?
Minion: Me molto bene! Banana alegria! Yu?

You: me sleepy-sleepy, dormi now po ka?
Minion: Close eyes-o. Me guard banana!

You: poopaye, see yu presto
Minion: Poopaye, amiko! See yu presto!
```

## Iteration history and honest limitations

The first curriculum withheld entire lexical concepts and achieved only 33% behavioral relevance even though style was strong. The second covered supported vocabulary but did not teach prefix invariance and reached 67%. The final curriculum used intent-balanced neutral lead-ins while keeping complete test strings disjoint; it reached 99% on the fixed test.

This high score measures interpolation over 25 intentionally narrow conversational intents, not general language understanding. In a separate manual probe, the compound prompt `I'm feeling sad and I could use a hug` produced a malformed greedy reply even though the supported atomic prompts `I'm feeling sad` and `Could you cheer me up?` worked. Arbitrary factual, compositional, or unfamiliar requests remain out of scope.

## Research basis

Pierre Coffin has explained that recognizable fragments in Spanish, French, Italian, and Indonesian help audiences understand otherwise invented speech. Commonly documented phrases provide a small canonical vocabulary, while the rest of this project's dialect and every training conversation were authored for this repository. No film subtitle or screenplay corpus was used.

- ITV interview with Pierre Coffin: <https://www.itv.com/news/westcountry/2017-09-21/despicable-me-director-explains-how-the-minions-got-their-voice>
- AP interview on the improvised, melodic nature of Minionese: <https://apnews.com/article/65f236dec8a3798516b306bf161eae95>
- Common phrase reference: <https://www.cineworld.co.uk/blog/learn-to-speak-minionese/>
- MLX documentation: <https://ml-explore.github.io/mlx/build/html/index.html>

