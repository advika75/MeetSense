# Takeaway Extraction Evaluation

Labels: `data/labels/meeting1.json`
Matching: timestamp window +/-30s OR fuzzy text similarity >= 80

| Category | rule_based Precision | rule_based Recall | rule_based F1 | llm Precision | llm Recall | llm F1 | llm_verify Precision | llm_verify Recall | llm_verify F1 | hybrid Precision | hybrid Recall | hybrid F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Decisions | 0.0 | 0.0 | 0.0 | 0.375 | 0.5 | 0.429 | 0.429 | 0.5 | 0.462 | 0.6 | 0.5 | 0.545 |
| Action items | 0.086 | 1.0 | 0.158 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Deadlines (strict) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.5 | 0.667 |
| Deadlines (lenient) | 0.0 | 0.0 | 0.0 | 1.0 | 0.5 | 0.667 | 1.0 | 0.5 | 0.667 | 1.0 | 0.5 | 0.667 |

## rule_based

### False positives
- **Decisions**
  - 1141.6s — Yeah. We decided also the administration stuff and. Sure.
- **Action items**
  - 46.0s — next Tuesday. And since we don't really have any other time, we have to make these decisions
  - 111.4s — that we have to put somewhere are like I said, the photocopier, which is really big, the
  - 144.0s — we have now, but just one shelf that's attached to the wall. And the people that we have to
  - 214.3s — Do we know if there will be a lot of people coming across the whole in terms of security
  - 236.5s — then it should be okay to leave them in the hall.
  - 316.2s — Yeah. And do we need a room like a reading room?
  - 325.6s — So we should leave a space for that.
  - 473.9s — So probably she should get the facts, right? Yeah. Yeah. That makes sense.
  - 510.6s — And then I think the main discussion we should have is if we want to make group of
  - 572.1s — you discuss things. Maybe I don't know if it's for other people who's not working in the same
  - 579.4s — project. Yeah. But we will have this reading room. Yeah. That's true. Yeah. That's true.
  - 584.2s — Will it be dedicated for reading and for studying or can we have meetings that can reserve it for
  - 606.1s — well the problem is you have to go across the building and it adds some overhead every time you
  - 611.4s — want to have some short meeting and discuss some issues. Maybe we shouldn't call it.
  - 631.1s — Because there's still the cafeterias that you can use for informal. Yeah. The other thing we should
  - 657.0s — very nice to have to share on a face with somebody like us. I think a lot of social life and
  - 704.9s — where your schedule is completely don't overlap and then you do schedule office hours when
  - 808.1s — or something. I think we should reserve a little bit space for them to. So yeah that's a good
  - 874.1s — So for the reading room we will have a two person room. Yeah I think so. Everybody agrees on that?
  - 880.2s — One person will be too small anyways. How about this room? The one on the bottom? Yeah the bottom
  - 905.6s — the same one between the big. Oh okay. Yeah sorry I should have explained that. But
  - 1147.8s — So we exclude the the thing we are quite sure and then we discuss there.
  - 1152.0s — I was thinking of maybe we should assign Gisella into this three person office because we will
  - 1214.4s — that she needs to be near the mailbox and the printer and.
  - 1255.5s — It should be a two people room and not three people room.
  - 1310.2s — Okay. Maybe we should write this down.
  - 1380.6s — I mean that's quite important to keep a spare room in Gisella's office if we need more staff.
  - 1429.2s — No, they need to go in the same room.
  - 1660.3s — I'll put her in gray, I guess.
  - 1708.9s — No, I have to put some more.
  - 1737.4s — We can come back and finish it.
  - 1738.7s — Because if we just sit here and discuss,

### Missed items
- **Decisions**
  - 593.0s — The reading room can also be used for meetings if it is empty (not reserved, no one reading)
  - 874.0s — The reading room will be a two-person room
  - 1029.0s — The middle two-person room is chosen as the reading room
  - 1311.0s — The photocopier and the network printer will go next to the mailbox
  - 1432.0s — Gisella is placed in the corner (southwest) office
  - 1652.0s — Jazeela is removed from the placement list because she already has a room
- **Deadlines (strict)**
  - 39.0s — Administration wants the office and equipment placement by November 2nd (next Tuesday)
  - 18.0s — The group moves to the new building in three weeks
- **Deadlines (lenient)**
  - 39.0s — Administration wants the office and equipment placement by November 2nd (next Tuesday)
  - 18.0s — The group moves to the new building in three weeks

## llm

### False positives
- **Decisions**
  - 46.0s — The group agrees to make decisions about office assignments and equipment placement by November 2nd.
  - 766.6s — One person can share an office with three people, making more noise and having more computers.
  - 785.0s — The three-person room can be used as a reading room with two people each, or as a temporary space for stagia or others.
  - 1311.4s — Agree that the photo copier and the network printer goes next to the mailbox
  - 1670.6s — Keep the current arrangement for the new building.
- **Action items**
  - 1728.3s — Take a break and think about the arrangement.

### Missed items
- **Decisions**
  - 874.0s — The reading room will be a two-person room
  - 1029.0s — The middle two-person room is chosen as the reading room
  - 1432.0s — Gisella is placed in the corner (southwest) office
- **Action items**
  - 185.0s — Present the plan to the University administrators
  - 193.0s — Come up with a placement plan, with a quick presentation arguing why it works best
  - 288.0s — Think over the concepts during lunch and come back later in the day with specific proposals for where to put people
- **Deadlines (strict)**
  - 39.0s — Administration wants the office and equipment placement by November 2nd (next Tuesday)
  - 18.0s — The group moves to the new building in three weeks
- **Deadlines (lenient)**
  - 18.0s — The group moves to the new building in three weeks

## llm_verify

### False positives
- **Decisions**
  - 46.0s — The group agrees to make decisions about office assignments and equipment placement by November 2nd.
  - 766.6s — One person can share an office with three people, making more noise and having more computers.
  - 1311.4s — Agree that the photo copier and the network printer goes next to the mailbox
  - 1670.6s — Keep the current arrangement for the new building.
- **Action items**
  - 1728.3s — Take a break and think about the arrangement.

### Missed items
- **Decisions**
  - 874.0s — The reading room will be a two-person room
  - 1029.0s — The middle two-person room is chosen as the reading room
  - 1432.0s — Gisella is placed in the corner (southwest) office
- **Action items**
  - 185.0s — Present the plan to the University administrators
  - 193.0s — Come up with a placement plan, with a quick presentation arguing why it works best
  - 288.0s — Think over the concepts during lunch and come back later in the day with specific proposals for where to put people
- **Deadlines (strict)**
  - 39.0s — Administration wants the office and equipment placement by November 2nd (next Tuesday)
  - 18.0s — The group moves to the new building in three weeks
- **Deadlines (lenient)**
  - 18.0s — The group moves to the new building in three weeks

## hybrid

### False positives
- **Decisions**
  - 39.6s — Make decisions by November 2nd
  - 1141.6s — Decide on the administration stuff
- **Action items**
  - 1747.2s — Take a break

### Missed items
- **Decisions**
  - 593.0s — The reading room can also be used for meetings if it is empty (not reserved, no one reading)
  - 1311.0s — The photocopier and the network printer will go next to the mailbox
  - 1652.0s — Jazeela is removed from the placement list because she already has a room
- **Action items**
  - 185.0s — Present the plan to the University administrators
  - 193.0s — Come up with a placement plan, with a quick presentation arguing why it works best
  - 288.0s — Think over the concepts during lunch and come back later in the day with specific proposals for where to put people
- **Deadlines (strict)**
  - 18.0s — The group moves to the new building in three weeks
- **Deadlines (lenient)**
  - 18.0s — The group moves to the new building in three weeks

## Runtime and peak RAM (qwen2.5:3b, Ollama, seed 42, temperature 0)

| Extractor | Runtime | Peak Ollama RSS |
|---|---|---|
| rule_based | 0.0s | n/a |
| llm | 228.5s | 2174 MB |
| llm_verify | 263.8s | 2195 MB |
| hybrid (run 1 / run 2) | 254.9s / 270.7s | ~2.05 GB (run 2; run 1's sampler read 478 MB) |

Hybrid run 1 and run 2 produced identical decisions, action items, deadlines and open questions.
