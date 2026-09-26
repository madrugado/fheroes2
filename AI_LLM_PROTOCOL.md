# Протокол событий ИИ для внешней LLM-модели

Дизайн-документ. Цель: дать внешней модели (LLM) доступ к состоянию игры и решениям
штатного ИИ в виде **структурированных полей** (без картинок), чтобы в перспективе
заменить или дополнить алгоритмический ИИ.

## Принципы

1. **Честность наблюдения**: модель получает ровно ту информацию, которую видит
   штатный ИИ. Туман — только разведанная территория (`tile.isFog(myColor)`);
   армии врагов — в виде strength-оценок из кэша `Planner::_enemyArmies`, а не
   точного состава. Никаких данных, которых у алгоритмического ИИ нет.
2. **Провод — JSON-lines**, компактные ключи, enum'ы числами. Рендеринг для модели
   (CSV-таблицы, построчные строки сетки и т.п.) — задача Python-sidecar, не C++.
3. **Append-only, одна строка = одно событие.** Файл можно читать на лету
   (`tail -f`), обрыв строки не портит остальное.
4. **Никаких зависимостей**: ручной JSON-эмиттер (~50 строк), без библиотек.
5. **Не сериализуется в сейвы**, не влияет на игровую логику (только запись).

## Канал

- Новый модуль: `src/fheroes2/ai/ai_log.h` + `ai_log.cpp`, пространство имён `AILog`.
- Включение через переменную окружения (CLI-парсинга в проекте нет):
  `FHEROES2_AI_LOG=/tmp/fh2_ai.jsonl ./fheroes2`. Не задана — эмиттер выключен,
  стоимость ≈ один `if`.
- API: `AILog::event("hero_target", {...})` — билдит строку вручную (экранирование
  строк, целые/дробные числа), пишет в буферизованный `FILE*`, flush по строке.
- `battleId` — статический счётчик в `AILog`, инкремент на `battle_start`.

## Соглашения о полях

| Ключ | Смысл |
|------|-------|
| `ev` | тип события |
| `t`  | день/ход (`world.CountDay()`) |
| `p`  | цвет игрока: `R`,`B`,`G`,`Y`,`O`,`P` (RED..PURPLE), `N` = neutral |
| `res`| ресурсы позиционным массивом `[wood, mercury, ore, sulfur, crystal, gems, gold]` |
| `h`  | id героя (`Heroes::GetID()`) |
| `i`  | тайл (индекс в `world`) |
| `obj`| `MP2::MapObjectType` (число) |
| `mon`| id монстра, `q` — количество |
| `bid`| id боя |
| `u`  | `Battle::Unit::GetUID()` |
| `b`  | `BuildingType` (число) |

Словарь id → имена не пишется в каждую строку: sidecar держит статическую таблицу
(MP2-объекты, монстры, здания — стабильные константы движка); при `game_start`
опционально пишется событие `legend` с используемыми словарями.

## События и точки вставки

### Снимок королевства — `turn_start`
Пишется в начале каждого AI-хода.
Вставка: `ai_planner_kingdom.cpp`, `Planner::KingdomTurn` (рядом с DEBUG_LOG про
"starts the turn", ~строка 678).
```json
{"ev":"turn_start","t":14,"p":"R","res":[12,3,9,2,4,5,3210],
 "castles":[{"n":"Ironfist","i":401,"threat":true}],
 "heroes":[{"id":3,"i":234,"mp":1800,"mmp":1800,"str":4321.5}],
 "diff":4}
```
Поля героя: `id`, позиция `i`, очки хода `mp`/`mmp`, сила армии `str`
(`Army::GetStrength()`), уровень `lvl`, статы `a/d/p/k`, вторичные навыки — по
необходимости (эволюционируемо, схема append-only по полям).

### Роль героя — `hero_role`
Вставка: `ai_planner_hero.cpp`, `setHeroRoles()` (вызывается из `KingdomTurn`).
```json
{"ev":"hero_role","t":14,"p":"R","h":3,"role":"fighter"}
```
`role`: `fighter` / `courier` / `scout` (уже вычисляется штатным кодом).

### Решение о цели — `hero_target`
Главная точка интеграции: сюда в будущем прилетит выбор LLM.
Вставка: `ai_planner_hero.cpp`, `Planner::HeroesTurn`, после выбора
`bestTargetIndex`/`maxPriority` (~строки 3145–3157).
```json
{"ev":"hero_target","t":14,"p":"R","h":3,"from":234,"to":512,"obj":18,"v":87.3,"d":4}
```
`obj` — тип объекта на тайле цели, `v` — приоритет из `getPriorityTarget`,
`d` — дистанция пути.

### Действия героя на карте — `visit`
Единая точка: `ai_hero_action.cpp`, `AI::HeroesAction(hero, dst_index)` (~1973) —
диспетчер всех визитов объектов. Плюс уточняющие события из соседних веток:
```json
{"ev":"visit","t":14,"p":"R","h":3,"i":512,"obj":18}
{"ev":"army_join","t":14,"p":"R","h":3,"mon":12,"q":20,"cost":1000}
{"ev":"castle_capture","t":14,"p":"R","h":3,"i":401,"from":"B"}
{"ev":"hero_meet","t":14,"p":"R","h":3,"h2":7}
```
Якоря — существующие DEBUG_LOG: "visits", "attacks", "join", "captures"
(`ai_hero_action.cpp`, ~45 точек).

### Движение героя — `hero_move`
Вставка: `ai_hero_action.cpp`, `AI::HeroesMove` — по завершении движения:
```json
{"ev":"hero_move","t":14,"p":"R","h":3,"from":234,"to":512,"spent":1200}
```

### Найм героя — `hero_recruit`
Вставка: `ai_planner_kingdom.cpp`, `purchaseNewHeroes()` → `recruitHero()`.
```json
{"ev":"hero_recruit","t":14,"p":"R","h":9,"i":401}
```

### Развитие замка — `castle_build`, `troops_hire`
Вставки: `ai_common.cpp` `AI::BuildIfPossible()` (успех постройки);
`ai_planner_castle.cpp` (~321, найм войск).
```json
{"ev":"castle_build","t":14,"p":"R","i":401,"b":23}
{"ev":"troops_hire","t":14,"p":"R","i":401,"mon":12,"q":10}
```

### Начало боя — `battle_start` (снимок боя)
Вставка: `battle_arena.cpp` (~438, рядом с `BattlePlanner::battleBegins()`).
```json
{"ev":"battle_start","bid":1,"t":14,"att":"R","def":"B","castle":false,
 "armies":{
  "att":{"hero":{"n":"Roland","a":5,"d":4,"p":3,"k":2,"sp":20},
         "stacks":[{"u":11,"mon":12,"q":42,"i":13,"sp":5}]},
  "def":{"hero":{"n":"...","a":3,"d":3,"p":1,"k":1,"sp":8},
         "stacks":[{"u":21,"mon":18,"q":30,"i":85,"sp":4}]}},
 "obstacles":[30,31,44],
 "covr":2}
```
Позиции стеков — индексы клеток `Board` 11×9 (0..98); у широких юнитов добавить
`ti` (tail index). Стрельбы — `shots`.

### Действие в бою — `battle_action`
Вставка: `battle_arena.cpp` (~508) сразу после
`AI::BattlePlanner::Get().BattleTurn(*this, *_currentUnit, actions)` — единая
точка, туда же в будущем встанет ответ LLM. Пишем команды как есть:
```json
{"ev":"battle_action","bid":1,"u":11,"act":"ATTACK","args":[11,21,45,46,3],"v":12.3}
{"ev":"battle_action","bid":1,"u":11,"act":"SPELLCAST","args":[1,85],"v":33.1}
```
`act` — `CommandType` (`MOVE/ATTACK/SPELLCAST/MORALE/CATAPULT/TOWER/RETREAT/
SURRENDER/SKIP`), `args` — параметры `Battle::Command`, `v` — оценка решения,
когда она известна (атака/заклинание).

### Конец боя — `battle_end`
Вставка: `battle_main.cpp`, где формируется `Battle::Result`.
```json
{"ev":"battle_end","bid":1,"winner":"R","rounds":7,
 "att":{"str0":4321.5,"str1":2100.0},"def":{"str0":3800.0,"str1":0.0}}
```
Сила до/после — из `analyzeBattleState`-подобных данных `BattlePlanner`
(`_myArmyStrength`/`_enemyArmyStrength` на старте) и армий в конце.

### Конец игры — `game_end`
Вставка: `game_startgame.cpp`, `gameResult.checkGameOver()`.
```json
{"ev":"game_end","t":97,"winner":"R","reason":"ENEMY_WILL_WIN"}
```

## Чего НЕ пишем

- Тайлы в тумане (`isFog(myColor)`) и объекты на них.
- Точный состав чужих армий вне правил разведки — только strength-оценки.
- Состояние интерфейса/графики, seed'ы ГСЧ (не дублируем deterministic-механику).

## Sidecar (Python)

1. Читает JSONL инкрементально, накапливает состояние.
2. Рендерит «брифинг» в компактный текст (терse-таблицы, построчные строки боевой
   сетки) — формат подгоняется под модель независимо от C++.
3. Отправляет LLM, получает решение, валидирует против легальных вариантов
   (список клеток для хода, список целей для атаки строится sidecar'ом из
   состояния), при невалидном ответе — fallback на штатный ИИ.
4. Фазы: **(1)** пассивный лог + оффлайн-анализ партий; **(2)** LLM решает бои
   (`battle_action` принимает ответ); **(3)** LLM выбирает цели героев
   (`hero_target` принимает ответ); **(4)** полный `KingdomTurn`.

## Порядок работ

1. `AILog` + `turn_start`, `hero_target`, `visit`, `battle_start`,
   `battle_action`, `battle_end` — минимальный набор для наблюдения.
2. Проверка на auto playtest (AI-vs-AI) — целостность лога, размер на ход.
3. Sidecar + брифинг; оффлайн-прогоны.
4. Обратный канал (бой), затем (стратегия).
