# Stockfish Features We're NOT Currently Extracting

## 1. 🎯 **Principal Variations (FREE!)**
- **Current**: Only first move of best line
- **Available**: Full 15-25 move sequences for each variation
- **Value**: See full plans, compare model's intended continuations

## 2. 📊 **Search Statistics**
- **Depth reached**: Actual vs requested (shows position complexity)
- **Selective depth**: How deep tactics were calculated
- **Nodes searched**: Computational effort (2M+ nodes)
- **Nodes per second**: Search efficiency
- **Time used**: Actual vs limit

## 3. 💾 **Engine Performance**
- **Hash table usage**: Memory efficiency (0-100%)
- **Tablebase hits**: Perfect endgame knowledge
- **Multi-PV scores**: Full comparison of top N moves
- **Effective branching factor**: Move ordering quality

## 4. 🎮 **Game Understanding**
- **Ponder move**: Expected opponent response
- **Threat detection**: What happens if we pass
- **Sacrifice detection**: Material vs positional trades
- **Critical positions**: Where precision matters most

## 5. 📈 **Position Characteristics**
- **Sharpness**: How much move quality matters
- **Complexity**: Evaluation spread between moves
- **Phase detection**: Opening/middlegame/endgame
- **Material imbalance**: Compensation detection

## 6. 🔍 **Advanced Analysis** (Requires extra passes)
- **Best response sequences**: For each player move
- **Blunder alternatives**: What they should have played
- **Defense resources**: How to hold difficult positions
- **Winning technique**: Converting advantages

## 7. ⚙️ **Configuration Options**
- **Contempt**: Playing style (aggressive/solid)
- **Skill Level**: Weakened play for training
- **Move Overhead**: Time management
- **Analysis Contempt**: Objective vs practical

## What This Means For Your Thesis

### Currently Getting (per move):
```json
{
  "eval_before": 149,
  "eval_after": 140,
  "best_move": "b3",
  "centipawn_loss": 9,
  "alternatives": ["Bd3", "a3"]
}
```

### Could Be Getting (same computation time):
```json
{
  "eval_before": 149,
  "eval_after": 140,
  "best_move": "b3",
  "centipawn_loss": 9,
  "depth_reached": 20,
  "selective_depth": 28,
  "nodes_searched": 2346452,
  "time_ms": 1655,
  "principal_variation": ["b3", "O-O", "Bb2", "b6", "Be2", "Bb7", "O-O", "Rc8", ...],
  "ponder_move": "O-O",
  "alternatives": [
    {"move": "Bd3", "eval": 148, "pv": ["Bd3", "O-O", "O-O", ...]},
    {"move": "a3", "eval": 140, "pv": ["a3", "a5", "Bd3", ...]}
  ],
  "position_sharpness": 0.85,
  "hash_usage": 76.4,
  "phase": "middlegame"
}
```

### Valuable for Mechanistic Interpretability:
1. **Full PVs**: Compare model's multi-move plans vs optimal
2. **Search effort**: Which positions require more computation
3. **Position sharpness**: Where models must be precise
4. **Phase-specific performance**: Opening vs endgame understanding
5. **Threat awareness**: Can models see opponent's ideas?

Want me to upgrade the annotator to capture all this rich data? 