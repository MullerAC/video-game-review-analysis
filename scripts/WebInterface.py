from scripts import predictions

def get_examples():
    examples = {'twitter': ['#stateofplay',
                            '#pokemonpresents',
                            '#MonsterHunter'
                           ],
                'reddit': ['https://www.reddit.com/r/nintendo/comments/lm6obv/project_triangle_strategy_announcement_trailer/',
                           'https://www.reddit.com/r/nintendo/comments/lru33p/the_animal_crossing_new_horizons_free_update_is/',
                           'https://www.reddit.com/r/Games/comments/ls6qlh/bravery_default_ii_review_thread/',
                           'https://www.reddit.com/r/Games/comments/lw3wtu/aliens_fireteam_official_announcement_trailer/'
                          ]
               }
    
    return examples

def get_web_output(source, limit=1000, samples=5):
    df = predictions.get_predictions(source, limit)
    value_counts = df.positive.value_counts(normalize=True)
    pos_percentage = round(value_counts[True]*100)
    neg_percentage = round(value_counts[False]*100)
    pos_samples = df[df['positive']].sample(5)['review'].tolist()
    neg_samples = df[~df['positive']].sample(5)['review'].tolist()
    
    return {'pos_percentage': pos_percentage,
            'neg_percentage': neg_percentage,
            'pos_samples': pos_samples,
            'neg_samples': neg_samples}