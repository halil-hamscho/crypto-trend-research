# Timestamp proofs

The protocol was timestamped with [OpenTimestamps](https://opentimestamps.org)
on **2026-09-24**, before `basket.py` existed — there was no engine that could
have produced a strategy return.

Each `.ots` file proves that the exact bytes of its document existed before a
given Bitcoin block. It proves *existence in time*, nothing else: not
authorship, not that the result was unknown. The substantive protection is the
protocol's own specificity and its disclosures (Part 1 of the protocol).

## 2026-09-30, fourth stamp: the pre-registration with Exception 1

Only `pre_registration_2026-09-22.md` changed (Part 10 gained **Exception 1**, the
2026-06-24 funding settlement). Its new SHA-256 is `97a1dbd31b59ca232af1ea3c1da123566e0e463b97ff5d57aa05e9478ee4af5b`; the proof next to it is
the current one. The other three files keep the third stamp below. The previous
version of this file and its proof are kept unchanged in `ots_2026-09-30_third_stamp/`.
The same caveat applies as for the third stamp: this shows existence from now on and
does **not** restore pre-registration status. Calendar attestations only; run
`ots upgrade` later.

## 2026-09-30, third stamp: the documents as revised (Amendments 4-6, Correction 1)

**These are the current hashes. Read the caveat before citing this proof.**

| file | SHA-256 |
|---|---|
| `pre_registration_2026-09-22.md` | `087235ca95523f1320e8afd4db902774c21f346d82bca8dbfd7c57878ccc0743` |
| `pre_registration_amendment_2026-09-23.md` | `d9c16e74cb947bb8532041418f95c650a60fe1268e3d81bca47054024a470fa7` |
| `frozen_config.yaml` | `7c23eb9121b5a1039695a5f4aee15721c4233e00d7143fb7cc466c6b14eb03ac` |
| `design_rationale.md` | `5903f3800de2a09b58267c661b0ddc22b149d6de5307f8fb3449c3025d12fc41` |

What changed since the previous stamp: Amendments 4 and 5 (2026-09-24/25), which
were committed but never stamped; Amendment 6 (2026-09-30, scored window ends
2026-08-30); and Correction 1 (2026-09-30, the LUNA membership statement, which
described an earlier membership rule, not the registered one).

**What this proof does and does not establish.** It shows these exact bytes existed
before the Bitcoin block it will be confirmed in. It does **not** restore
pre-registration status: Amendment 6 and Correction 1 were written after a
sanity run of the benchmark had been observed (net Sharpe 0.34), and Amendments 4
and 5 after the earlier stamp. The original, pre-amendment versions and their
proofs are kept unchanged in `ots_2026-09-24_amendment3/`; their hashes are in the
table of the second stamp below. The git history holds every intermediate version.

The proofs carry calendar attestations only. Run `ots upgrade <file>.ots` after a
few hours, then commit the upgraded files.

## 2026-09-24, second stamp — the protocol as amended through Amendment 3

These are the authoritative hashes. The first-round proofs below remain valid for
the pre-Amendment-3 bytes and are kept in `ots_v1/`.

| file | SHA-256 |
|---|---|
| `pre_registration_2026-09-22.md` | `5a848382c238daf4aa9d120bfa2a4cce6b89363bdf39de2517ae743604fffb67` |
| `pre_registration_amendment_2026-09-23.md` | `5bd1234912e62b175c5bba4877c49d7e57560a80c352875353c850245568064f` |
| `frozen_config.yaml` | `80fde9d568c1edd3ae55619cf4a18f6bb585164c40ef9403ef98ea19a9de2f46` |
| `design_rationale.md` | `333bea4fe4df9a75bf3057fedb40808a7d742b51a74e882c28dc71f18c13e40f` |

Frozen commit: `9ae923a`.

## 2026-09-24, first stamp (Amendments 1–2)

| file | SHA-256 |
|---|---|
| `pre_registration_2026-09-22.md` | `7e8e0079c3e224c758fcf7988b8ab5db5cc44eb3f68580a154295f310be93b32` |
| `pre_registration_amendment_2026-09-23.md` | `5985e84247c8bd7f9bb08e82361afa53f7380ec0e544e2d1aa34a5afae63e589` |
| `prereg_support/frozen_config.yaml` | `8b9336fa0588e4fe66cfb7e1ad3897a47198e468dbd01b5b3a4eb9cde08d6e45` |

Editing any of those files breaks its proof; a changed file needs a new stamp,
recorded as another dated row here. (The proofs next to the documents are now those
of the third stamp; the second and first stamps' proofs are in `ots_2026-09-24_amendment3/` and `ots_v1/`.)

## Verifying

```bash
pip install opentimestamps-client
ots verify docs/september/pre_registration_2026-09-22.md.ots
```

The proofs currently carry calendar attestations. Bitcoin confirmation takes a
few hours; run `ots upgrade <file>.ots` afterwards to fold the block attestation
into the proof, then commit the upgraded files.

On this machine, Python needs the system CA bundle for the calendar servers:
prefix commands with `SSL_CERT_FILE=/etc/ssl/cert.pem`.

## Still open

An OSF registration (embargoed, since the repository is private) when the paper
is written. Zenodo's GitHub integration cannot be used: it archives public
repositories only.
