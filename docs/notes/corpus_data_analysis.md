# Corpus data and tokenizer analysis

## Corpus snapshot

The core corpus uses six DOLLMA components: `anl-news`, `azwiki`, `elite-blogs`, `elite-books`, `eqanun`, and `mediocore-books`. The frozen raw inventory contains 14 core parquet shards (1,849,066,393 compressed bytes). `translated-enwiki` remains outside the core corpus. `bhos` is also excluded because its source role is still unresolved.

The six core sources contain 8,227,654 raw records and 3,012,376,236 Unicode characters. Books II (`mediocore-books`) contributes most document rows, while news contributes many more characters per document. A proportional document sample would therefore be dominated by short Books II fragments.

| Source | Group | Raw docs | Raw chars | Short removed | Short removed % | Exact removed | Near removed | Retained | 16K tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anl-news | News | 236,443 | 1,286,065,173 | 189 | 0.08% | 11,845 | 2,524 | 221,885 | 237,559,904 |
| azwiki | Native Wikipedia | 129,433 | 312,205,937 | 0 | 0.00% | 53 | 217 | 129,163 | 74,610,988 |
| elite-blogs | Blogs | 755 | 6,534,325 | 16 | 2.12% | 0 | 4 | 735 | 1,528,244 |
| elite-books | Books | 104 | 33,074,624 | 0 | 0.00% | 1 | 6 | 97 | 7,950,165 |
| eqanun | Laws | 53,656 | 348,973,113 | 5 | 0.01% | 2,756 | 1,087 | 49,808 | 64,643,900 |
| mediocore-books | Books | 7,807,263 | 1,025,523,064 | 1,873,780 | 24.00% | 129,825 | 11,415 | 5,792,243 | 219,427,536 |

Raw document lengths vary sharply by component. The figures below are Unicode-character counts before cleaning.

| Source | Median chars | P95 chars | P99 chars |
| --- | --- | --- | --- |
| anl-news | 3,873 | 15,110 | 24,698 |
| azwiki | 1,244 | 7,834 | 20,384 |
| elite-blogs | 3,446 | 38,455 | 60,346 |
| elite-books | 267,352 | 662,826 | 1,190,403 |
| eqanun | 1,763 | 26,069 | 65,443 |
| mediocore-books | 103 | 326 | 520 |

## Cleaning and duplicate accounting

Normalization uses NFC, converts CRLF to LF, trims outer whitespace, removes unsafe control characters, and preserves Azerbaijani spelling and case. The 50-Unicode-letter filter removed 1,873,990 records. Of these, 1,873,780 came from Books II. This is a material filtering choice: it removes many short fragments and should be kept in mind when interpreting the final source mixture.

After length filtering, global canonical-text deduplication removed 144,480 records in 107,267 duplicate groups. No exact group spanned source labels. The exact-unique corpus contained 6,209,184 documents.

The near-duplicate stage evaluated 6,444,499 unique LSH candidates at the frozen 0.95 character-5-gram Jaccard threshold. It accepted 86,697 edges, forming 13,208 connected components over 28,461 documents. The implementation retains one deterministic representative per component, removing 15,253 documents. Every accepted graph edge satisfies direct Jaccard >= 0.95; transitive members of a component are not guaranteed to be pairwise above that threshold. The final retained corpus has 6,193,931 documents. All accounting identities reconcile.

An independent pre-experiment audit found that the first implementation used anchor-only expansion in LSH buckets above 200 documents. Its measured recall was 1.827% on the frozen audit sample. Before any model training, that shortcut was replaced with complete, chunked pair enumeration for all observed buckets. The same sample then captured 28,849 of 28,849 eligible true pairs. A second exhaustive check covered all 59 formerly problematic bucket events and captured 321,263 of 321,263 eligible pairs, with 0 misses.

Suspicious-text heuristics were used as audit flags, not broad language filters. The raw profile records markup-like text, unusual whitespace, repeated characters, replacement characters, line-break-heavy material, and a small number of null bytes. Short excerpts are stored in `data/metadata/raw_quality_profile.json`; flagged text was not automatically discarded merely for triggering a heuristic.

| Source | Flag events | Measured categories |
| --- | --- | --- |
| anl-news | 76 | control_character: 2, extremely_long: 12, html_xml_like_markup: 21, repeated_character: 22, unicode_replacement: 3, unusual_whitespace: 16 |
| azwiki | 68 | extremely_long: 22, high_line_break_density: 11, repeated_character: 35 |
| elite-blogs | 77 | high_line_break_density: 4, unicode_replacement: 1, unusual_whitespace: 72 |
| elite-books | 102 | extremely_long: 88, html_xml_like_markup: 3, repeated_character: 10, unusual_whitespace: 1 |
| eqanun | 3,351 | control_character: 67, extremely_long: 302, high_line_break_density: 1511, html_xml_like_markup: 15, null_byte: 44, repeated_character: 18, unicode_replacement: 16, unusual_whitespace: 1378 |
| mediocore-books | 1,591 | control_character: 164, html_xml_like_markup: 590, repeated_character: 535, unicode_replacement: 274, url_heavy: 28 |

## Split and leakage

The split is document-level and cluster-aware, using seed 2026 and the frozen 90/5/5 hash ranges.

| Split | Documents | 16K tokens incl. EOD | Unknown tokens |
| --- | --- | --- | --- |
| train | 5,574,885 | 544,498,912 | 2,712 |
| validation | 309,677 | 30,329,083 | 46 |
| test | 309,369 | 30,892,742 | 115 |

| Source | Train docs | Train tokens | Val docs | Val tokens | Test docs | Test tokens |
| --- | --- | --- | --- | --- | --- | --- |
| anl-news | 199,587 | 213,954,688 | 11,217 | 12,049,657 | 11,081 | 11,555,559 |
| azwiki | 116,025 | 66,755,158 | 6,506 | 3,812,223 | 6,632 | 4,043,607 |
| elite-blogs | 665 | 1,406,264 | 33 | 60,108 | 37 | 61,872 |
| elite-books | 85 | 7,077,178 | 7 | 672,080 | 5 | 200,907 |
| eqanun | 44,789 | 57,785,971 | 2,471 | 2,776,802 | 2,548 | 4,081,127 |
| mediocore-books | 5,213,734 | 197,519,653 | 289,443 | 10,958,213 | 289,066 | 10,949,670 |

The repaired hard leakage audit passed. Document IDs, canonical hashes, and accepted duplicate-cluster IDs have zero cross-split intersections. All 86,697 accepted near-duplicate edges remain within one split. The independent exhaustive large-bucket audit found 0 retained true-near pairs crossing splits, and all 237 prerepair confirmed pairs were resolved.

## Tokenizer candidates

All three SentencePiece BPE candidates used the same 1,000,000-document train-only corpus in the same document-ID order. Its SHA-256 is `da0ff4b8209ab40e98afc96c71584a15defbd962d2f50e9b4f5ebc4e0a65a1d1`. Internal line breaks are projected to spaces for SentencePiece encoding; the canonical processed text is unchanged.

The table below uses the same 100,000 train documents for every candidate. Fertility is SentencePiece tokens divided by approximate whitespace words. These BPE pieces are not interpreted as morphemes.

| Vocabulary | Tokens/word | Chars/token | UNK / token denominator | UNK rate | Docs with UNK | Model bytes | Vocab bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8,000 | 1.9212 | 4.0862 | 1 / 11,132,227 | 8.983e-08 | 1 | 129,151 | 119,717 |
| 16,000 | 1.6743 | 4.6888 | 1 / 9,701,653 | 1.031e-07 | 1 | 277,746 | 266,309 |
| 32,000 | 1.5086 | 5.2038 | 1 / 8,741,364 | 1.144e-07 | 1 | 586,192 | 574,754 |

The 16K source-wise audit used the same shared train sample:

| Source | Audited docs | 16K tokens/word |
| --- | --- | --- |
| anl-news | 3,612 | 1.5753 |
| azwiki | 2,063 | 1.8396 |
| elite-blogs | 14 | 1.7242 |
| elite-books | 1 | 1.7764 |
| eqanun | 864 | 1.5160 |
| mediocore-books | 93,446 | 1.7824 |

Long and suffix-rich words were inspected as tokenization examples, not as evidence of morphological analysis:

| Word | 16K pieces | Piece count |
| --- | --- | --- |
| CənabPrezidenthələötənilinyekunlarınışərhedərkənbütündünyanıbürüyəniqtisadiböhranlarınölkəmizdəqarşısınınalınmasınınyollarınıdagöstərərəkbizinikbinolmağaruhlandırdı | ▁Cənab · P · rezident · h · ələ · ö · tən · ilin · y · ek · un · ların · ış · ər · he · dər · kən · b · ütün · d · üny · anıb · ür · ü · yən · iqtisadi · b · öh · ran · ların · öl · kə · mizdə · qar · şı · sının · alın · masının · yol · larını · da · gö · stər · ərək · b · izin · ik · bin · ol · mağ · ar · uh · landır · dı | 54 |
| təsərrüfatgöstəricilərivədigərməqamlarbarədəZaqatalaRayonİcraHakimiyyətibaşçısınınDağlıkəndinzibatiərazidairəsiüzrənümayəndəsiÖmərQarayevdahagenişməlumatlarverir | ▁təsərrüfat · gö · stər · icilər · iv · ədi · gər · m · əq · am · lar · bar · ədə · Z · aq · at · ala · R · ayon · İc · ra · H · akim · iyyət · ib · aş · çı · sının · Dağ · lı · kənd · in · zibati · ər · az · ida · ir · əsi · üz · r · ən · üm · ayəndə · si · Ö · mər · Qar · ayev · da · ha · gen · iş · m · əlum · atlar · ver · ir | 57 |
| tikinti-abadlıqvəquruculuqişlərihamımızısevindirirvəbuişləringörülməsindəazdaolsabizimnümayəndəliyindəəməyininolmasıcamaatımızıruhlandırır | ▁tikinti · - · ab · adlıq · v · əq · ur · uculuq · işləri · ham · ımızı · sev · in · dirir · v · əb · u · iş · lərin · gör · ül · məsində · az · da · ol · s · ab · iz · imn · üm · ayəndə · liyində · əm · əyinin · ol · ması · c · ama · at · ımız · ır · uh · landır · ır | 44 |
| eləcədəgənclərgöstərilənqayğıvədiqqətəgörəAzərbaycanRespublikasınınPrezidenticənabİlhamƏliyevəminnətdarlıqlarınıbildiriblər | ▁elə · cədə · gənc · lər · gö · stər · ilən · q · ay · ğı · və · diq · q · ətə · gö · rə · Azərbaycan · R · espublik · asının · P · rezident · ic · ənab · İl · ham · Əliyev · əmin · nət · darlıq · ların · ıb · il · dir · iblər | 35 |
| AzərbaycanRespublikasınınPrezidentiİlhamƏliyevinölkəninbütünistiqamətlərüzrəinkişafınagöstərdiyiqayğıvədiqqətgözönündədir | ▁Azərbaycan · R · espublik · asının · P · rezident · i · İl · ham · Əliyev · in · öl · k · ənin · b · ütün · ist · iq · amət · lər · üz · rə · in · kişaf · ına · gö · stər · diy · iq · ay · ğı · və · diq · q · ət · göz · ön · ündə · dir | 39 |
| AzƏrbaycanRespublikasınınPrezidentiyanındakütlƏviinformasiyavasitƏlƏrinininkişafınadövlƏtdƏstƏyifondununmaliyyƏyardımıilƏ | ▁Az · Ər · baycan · R · espublik · asının · P · rezident · iyan · ında · k · üt · l · Ə · vi · in · formasiya · v · asit · Əl · Ə · rinin · in · kişaf · ın · ad · öv · l · Ə · t · d · Ə · st · Ə · y · if · ond · unun · mal · iyy · Ə · yar · dım · ı · il · Ə | 46 |
| Zaqatalanınbirvaxtlarəlçatmazgörünənənucqardağkəndlərindədəmüasirləşməistiqamətlərindəxeyliişlərgörülmüşvəgörülməkdədir | ▁Zaqat · al · anın · bir · v · axt · lar · əl · çat · maz · gör · ün · ənən · uc · qar · dağ · kənd · lərində · də · mü · asir · ləşmə · ist · iq · amət · lərində · x · eyli · iş · lər · gör · ülmüş · və · gör · ül · məkdədir | 36 |
| BizhamımızbirlikdəAzərbaycanıdahadagözəlləşdirməkvəAzərbaycaninsanınınbütünproblemləriniçözməküçünəlbirolmalıyıq | ▁Biz · ham · ımız · bir · likdə · Azərbaycan · ı · da · h · ada · gözəl · ləşdirmək · və · Azərbaycan · ins · anının · b · ütün · pro · blem · lərini · ç · öz · mək · üç · ün · əl · bir · ol · malıyıq | 30 |

The preregistered 16K choice was retained. It produced 16,000 pieces, stable special-token IDs, sensible Azerbaijani round trips, and a train-audit unknown rate of 1.031e-07. Candidate rates use unknown SentencePiece tokens divided by all SentencePiece tokens in the shared 100,000-document audit sample. Full-corpus unknown counts use the final 16K tokenizer over each complete split and are reported as counts, not compared as if they shared the candidate-audit denominator. The lower fertility of 32K is expected from its larger vocabulary and is not, by itself, a reason to change the protocol.

Across the full retained corpus, the final 16K tokenizer yields 605,720,737 tokens including one `<eod>` per document. The model hash is `be05949c40afbe6031eee5678f49f5f49fad81cb1dd5fa8fb56c67f181222534`.

## Frozen 50M training corpus

The 50M selection uses train documents only, without replacement, with data seed 2026. Blogs contain only 1,406,264 unique train tokens against the requested 5,000,000. The 3,593,736-token shortage was redistributed across eligible native groups using the frozen weights.

| Group | Requested | Quota phase | Shortage | Final selected | Final share |
| --- | --- | --- | --- | --- | --- |
| Blogs | 5,000,000 | 1,406,264 | 3,593,736 | 1,406,264 | 2.81% |
| Books | 15,000,000 | 15,000,039 | 0 | 16,121,567 | 32.20% |
| Laws | 7,500,000 | 7,725,606 | 0 | 8,347,290 | 16.67% |
| Native Wikipedia | 10,000,000 | 10,000,175 | 0 | 10,748,666 | 21.47% |
| News | 12,500,000 | 12,503,436 | 0 | 13,439,100 | 26.84% |

Component provenance is retained inside the grouped Books quota:

| Source | Group | Selected tokens | Selected share |
| --- | --- | --- | --- |
| anl-news | News | 13,439,100 | 26.84% |
| azwiki | Native Wikipedia | 10,748,666 | 21.47% |
| elite-blogs | Blogs | 1,406,264 | 2.81% |
| elite-books | Books | 7,077,178 | 14.14% |
| eqanun | Laws | 8,347,290 | 16.67% |
| mediocore-books | Books | 9,044,389 | 18.07% |

The frozen manifest contains 277,027 unique documents and 50,062,887 tokens, an overshoot of 62,887 caused by preserving whole documents. Future training reads this one fixed sequence and stops after exactly 50,000,000 consumed model tokens. The boundary is document db05beef4e2c5dec4bf978a78afd788f579429852afea925852f7122b7608c36 at one-based sequence position 276,626: 4,185 of its 6,799 tokens are consumed, before its `<eod>`. Model initialization seeds do not change the subset or its order.

The downstream replay independently reproduced the tokenizer sample, all 6,193,931 document-token records, the selected IDs and order, and the exact boundary. Repository-internal references are relative. A simulated relocation to `C:/Research/azerbaijani-positional-encoding` resolved successfully; an external DOLLMA clone at a different location must be supplied through `AZ_PE_DOLLMA_ROOT`.

## Known limitations

The local DOLLMA README provides a dataset-level CC BY-NC-SA 4.0 declaration, but source-level licenses and revisions are not stated. The source snapshot was revalidated on 2026-08-28; the original acquisition date is not known. The Books I/Books II mapping is inferred from the published size descriptions and matching local component sizes; component identities remain separate in all artifacts. `bhos` still requires a source decision. The 50-letter rule disproportionately affects Books II fragments: it removed 24.00% of that source's raw rows. Connected-component closure can link endpoints below the direct 0.95 edge threshold; the sampled minimum endpoint similarity was about 0.820. Finally, language and OCR checks are heuristics, not verified language labels.
