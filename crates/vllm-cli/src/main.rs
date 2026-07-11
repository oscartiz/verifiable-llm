use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use vllm_core::chain::{SamplerConfig, SamplerMode};
use vllm_core::commit::{self, ModelCommitment, VerifyOutcome};
use vllm_core::transcript::{ChainCheck, Transcript};
use vllm_infer::engine::{self, GenerateRequest, Prompt};

/// Which Layer-3 proof system commits the logits and proves the decode step.
/// Both prove the same argmax statement; the salted 32-byte digest each
/// produces is folded into the chain identically, so the transcript/trace
/// format is backend-agnostic (see DECISIONS.md #17).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum ZkBackend {
    /// winterfell STARK: fast, succinct, but not formally zero-knowledge.
    #[default]
    Winterfell,
    /// halo2 (transparent IPA): formally zero-knowledge, slower. Requires the
    /// `zk-halo2` build feature.
    Halo2,
}

impl ZkBackend {
    /// The tag written into (and matched from) the decode-proof envelope.
    fn tag(self) -> &'static str {
        match self {
            ZkBackend::Winterfell => "winterfell",
            ZkBackend::Halo2 => "halo2",
        }
    }
}

/// Verifiable local LLM inference.
#[derive(Parser)]
#[command(name = "vllm", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Merkle-hash all tensors of a GGUF file into a model commitment.
    Commit {
        #[arg(long)]
        model: PathBuf,
        /// Output path (default: <model>.commitment.json).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Re-hash a GGUF file and check it against a model commitment.
    VerifyModel {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        commitment: PathBuf,
    },
    /// Generate text, committing every decode step into a hash chain.
    Generate(GenerateArgs),
    /// Re-check the hash chain of a generation transcript (no model needed).
    Replay {
        /// Transcript JSON produced by `vllm generate --commit`.
        #[arg(long)]
        commitment: PathBuf,
    },
    /// Derive the Fiat-Shamir challenge for a traced transcript.
    Challenge {
        /// Transcript JSON produced by `vllm generate --commit ... --trace`.
        #[arg(long)]
        commitment: PathBuf,
        /// Number of challenged cells.
        #[arg(short, long, default_value_t = 20)]
        k: u32,
        /// Verifier-chosen nonce; pick a fresh one after receiving the
        /// transcript to rule out prover grinding.
        #[arg(long, default_value = "")]
        nonce: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Answer a challenge from the local trace file (prover side).
    Respond {
        #[arg(long)]
        commitment: PathBuf,
        #[arg(long)]
        trace: PathBuf,
        #[arg(long)]
        challenge: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Produce a proof that a generated token is the argmax of its committed
    /// logit vector (transcript must be greedy + --prove-decode).
    ProveDecode {
        #[arg(long)]
        commitment: PathBuf,
        #[arg(long)]
        trace: PathBuf,
        /// Which generated step to prove.
        #[arg(long)]
        step: u32,
        #[arg(long)]
        out: PathBuf,
        /// Proof system; must match the one used at generation. `halo2` needs
        /// the `zk-halo2` build feature.
        #[arg(long, value_enum, default_value_t = ZkBackend::Winterfell)]
        backend: ZkBackend,
    },
    /// Verify an argmax decode proof against a transcript (no model, no
    /// trace, no GPU needed). The proof system is read from the proof file.
    VerifyDecode {
        #[arg(long)]
        commitment: PathBuf,
        #[arg(long)]
        proof: PathBuf,
    },
    /// Re-execute challenged layers on CPU and check the response.
    Verify {
        #[arg(long)]
        commitment: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        challenge: PathBuf,
        #[arg(long)]
        response: PathBuf,
        /// Max per-element |delta| for re-executed activations.
        #[arg(long, default_value_t = vllm_verify::DEFAULT_TOLERANCE)]
        tolerance: f32,
        /// Max per-element |delta| for recomputed logits.
        #[arg(long)]
        logits_tolerance: Option<f32>,
        /// Max mean |delta| per challenged cell (bounded-drift detector).
        #[arg(long, default_value_t = vllm_verify::DEFAULT_MEAN_TOLERANCE)]
        mean_tolerance: f32,
    },
}

#[derive(Args)]
struct GenerateArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    /// tokenizer.json; defaults to one next to the model file.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Write the generation transcript (chain commitment) here.
    #[arg(long)]
    commit: Option<PathBuf>,
    /// Reuse a model commitment instead of re-hashing the GGUF.
    #[arg(long)]
    model_commitment: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// Greedy decoding (default is top-p sampling).
    #[arg(long)]
    greedy: bool,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Fixed-point fractional bits for logit hashing.
    #[arg(long, default_value_t = engine::DEFAULT_LOGIT_FRAC_BITS)]
    frac_bits: u8,
    /// Run on CPU even if Metal is available.
    #[arg(long)]
    cpu: bool,
    /// Feed the prompt as-is (no llama3 chat template).
    #[arg(long)]
    raw: bool,
    /// Print a timing breakdown including commitment overhead.
    #[arg(long)]
    bench: bool,
    /// Write an activation trace file (required to answer v0.2 challenges).
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Commit each step's logits with a salted, circuit-friendly digest so the
    /// decode step can later be proved with `vllm prove-decode`. Requires
    /// --trace.
    #[arg(long, requires = "trace")]
    prove_decode: bool,
    /// Proof system for the decode commitment (with --prove-decode). `halo2`
    /// is formally zero-knowledge but needs the `zk-halo2` build feature.
    #[arg(long, value_enum, default_value_t = ZkBackend::Winterfell)]
    zk_backend: ZkBackend,
    /// Deterministic CPU backend: bit-exact re-execution, so challenges
    /// verify with tolerance zero. Slower than Metal; ~5 GB extra RAM.
    #[arg(long)]
    deterministic: bool,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Commit { model, out } => cmd_commit(&model, out),
        Command::VerifyModel { model, commitment } => cmd_verify_model(&model, &commitment),
        Command::Generate(args) => cmd_generate(args),
        Command::Replay { commitment } => cmd_replay(&commitment),
        Command::ProveDecode {
            commitment,
            trace,
            step,
            out,
            backend,
        } => cmd_prove_decode(&commitment, &trace, step, &out, backend),
        Command::VerifyDecode { commitment, proof } => cmd_verify_decode(&commitment, &proof),
        Command::Challenge {
            commitment,
            k,
            nonce,
            out,
        } => cmd_challenge(&commitment, k, &nonce, &out),
        Command::Respond {
            commitment,
            trace,
            challenge,
            out,
        } => cmd_respond(&commitment, &trace, &challenge, &out),
        Command::Verify {
            commitment,
            model,
            challenge,
            response,
            tolerance,
            logits_tolerance,
            mean_tolerance,
        } => cmd_verify(
            &commitment,
            &model,
            &challenge,
            &response,
            tolerance,
            logits_tolerance,
            mean_tolerance,
        ),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_str(&std::fs::read_to_string(path)?)
        .with_context(|| format!("parsing {path:?}"))
}

fn cmd_challenge(commitment: &Path, k: u32, nonce: &str, out: &Path) -> Result<()> {
    let transcript: Transcript = read_json(commitment)?;
    if transcript.replay_chain() != ChainCheck::Ok {
        bail!("transcript chain does not replay; refusing to challenge it");
    }
    transcript
        .check_trace_binding()
        .map_err(|e| anyhow::anyhow!("trace binding: {e}"))?;
    let challenge = vllm_core::protocol::make_challenge(&transcript, k, nonce)?;
    std::fs::write(out, serde_json::to_string_pretty(&challenge)?)?;
    let heads = challenge
        .cells
        .iter()
        .filter(|c| c.layer == transcript.trace.as_ref().unwrap().n_layers)
        .count();
    eprintln!(
        "derived {} challenge cells ({} block checks, {heads} head checks) -> {}",
        challenge.cells.len(),
        challenge.cells.len() - heads,
        out.display()
    );
    Ok(())
}

fn cmd_respond(commitment: &Path, trace: &Path, challenge: &Path, out: &Path) -> Result<()> {
    let transcript: Transcript = read_json(commitment)?;
    let challenge: vllm_core::protocol::Challenge = read_json(challenge)?;
    // Refuse challenges that are not the honest FS derivation.
    let expected = vllm_core::protocol::make_challenge(&transcript, challenge.k, &challenge.nonce)?;
    if expected.cells != challenge.cells
        || expected.final_chain != challenge.final_chain
        || expected.trace_root != challenge.trace_root
    {
        bail!("challenge is not the Fiat-Shamir derivation for this transcript");
    }
    let t0 = Instant::now();
    let mut reader = vllm_core::trace::TraceReader::open(trace)?;
    let response = vllm_core::protocol::build_response(&mut reader, &challenge)?;
    std::fs::write(out, serde_json::to_string(&response)?)?;
    eprintln!(
        "answered {} challenges in {:.2?} ({} KB) -> {}",
        response.items.len(),
        t0.elapsed(),
        std::fs::metadata(out)?.len() / 1024,
        out.display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_verify(
    commitment: &Path,
    model: &Path,
    challenge: &Path,
    response: &Path,
    tolerance: f32,
    logits_tolerance: Option<f32>,
    mean_tolerance: f32,
) -> Result<()> {
    let transcript: Transcript = read_json(commitment)?;
    let challenge: vllm_core::protocol::Challenge = read_json(challenge)?;
    let response: vllm_core::protocol::Response = read_json(response)?;
    let config = vllm_verify::VerifyConfig {
        tolerance,
        logits_tolerance: logits_tolerance.unwrap_or(tolerance),
        mean_tolerance,
    };
    let t0 = Instant::now();
    let report = vllm_verify::verify(model, &transcript, &challenge, &response, &config)?;
    for item in &report.items {
        eprintln!(
            "  cell (pos {:>4}, layer {:>2}) ok, max |delta| {:.3e}, mean {:.3e}",
            item.pos, item.layer, item.max_dev, item.mean_dev
        );
    }
    eprintln!(
        "OK: {} challenges verified in {:.2?}; worst max {:.3e} (tolerance {}), worst mean {:.3e} (mean tolerance {})",
        report.items.len(),
        t0.elapsed(),
        report.max_dev,
        config.tolerance,
        report.max_mean_dev,
        config.mean_tolerance
    );
    Ok(())
}

fn default_commitment_path(model: &Path) -> PathBuf {
    let mut name = model.file_name().unwrap_or_default().to_os_string();
    name.push(".commitment.json");
    model.with_file_name(name)
}

fn cmd_commit(model: &Path, out: Option<PathBuf>) -> Result<()> {
    let t0 = Instant::now();
    let commitment = commit::commit_gguf(model)?;
    let out = out.unwrap_or_else(|| default_commitment_path(model));
    std::fs::write(&out, serde_json::to_string_pretty(&commitment)?)?;
    eprintln!(
        "committed {} tensors in {:.2?}\nmodel root: {}\nwrote {}",
        commitment.tensors.len(),
        t0.elapsed(),
        commitment.root,
        out.display()
    );
    Ok(())
}

fn cmd_verify_model(model: &Path, commitment_path: &Path) -> Result<()> {
    let commitment: ModelCommitment =
        serde_json::from_str(&std::fs::read_to_string(commitment_path)?)
            .with_context(|| format!("parsing {commitment_path:?}"))?;
    if commitment.recompute_root() != Some(commitment.root) {
        bail!("commitment file is self-inconsistent: root does not match its own leaf hashes");
    }
    let t0 = Instant::now();
    match commit::verify_gguf(model, &commitment)? {
        VerifyOutcome::Ok => {
            eprintln!(
                "OK: model matches commitment root {} ({:.2?})",
                commitment.root,
                t0.elapsed()
            );
            Ok(())
        }
        VerifyOutcome::TensorMismatch { name } => bail!("MISMATCH: tensor {name:?} differs"),
        VerifyOutcome::StructureMismatch => bail!("MISMATCH: tensor table differs"),
        VerifyOutcome::RootMismatch { expected, actual } => {
            bail!("MISMATCH: root {actual} != committed {expected}")
        }
    }
}

fn load_or_build_model_commitment(args: &GenerateArgs) -> Result<ModelCommitment> {
    if let Some(path) = &args.model_commitment {
        let commitment: ModelCommitment = serde_json::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("parsing {path:?}"))?;
        if commitment.recompute_root() != Some(commitment.root) {
            bail!("model commitment {path:?} is self-inconsistent");
        }
        return Ok(commitment);
    }
    let t0 = Instant::now();
    let commitment = commit::commit_gguf(&args.model)?;
    eprintln!(
        "model root: {} (hashed {} tensors in {:.2?})",
        commitment.root,
        commitment.tensors.len(),
        t0.elapsed()
    );
    let cache = default_commitment_path(&args.model);
    if !cache.exists() && std::fs::write(&cache, serde_json::to_string_pretty(&commitment)?).is_ok()
    {
        eprintln!("cached model commitment at {}", cache.display());
    }
    Ok(commitment)
}

fn cmd_generate(args: GenerateArgs) -> Result<()> {
    let model_commitment = load_or_build_model_commitment(&args)?;

    let tokenizer = args.tokenizer.clone().or_else(|| {
        let candidate = args.model.with_file_name("tokenizer.json");
        candidate.exists().then_some(candidate)
    });
    if tokenizer.is_none() {
        bail!("no tokenizer.json found next to the model; pass --tokenizer");
    }

    let sampler = if args.greedy {
        SamplerConfig::greedy()
    } else {
        SamplerConfig {
            mode: SamplerMode::TopP,
            temperature: args.temperature,
            top_p: args.top_p,
            rng_seed: args.seed,
        }
    };

    let zk_commit = build_zk_commit(args.prove_decode, args.zk_backend)?;
    let req = GenerateRequest {
        model_path: args.model.clone(),
        tokenizer_path: tokenizer,
        prompt: Prompt::Text(args.prompt.clone()),
        raw: args.raw,
        max_new_tokens: args.max_tokens,
        sampler,
        logit_frac_bits: args.frac_bits,
        force_cpu: args.cpu,
        trace_path: args.trace.clone(),
        deterministic: args.deterministic,
        zk_commit,
    };

    let mut stdout = std::io::stdout();
    let mut stream = |s: &str| {
        let _ = stdout.write_all(s.as_bytes());
        let _ = stdout.flush();
    };
    let out = engine::generate(&req, &model_commitment, Some(&mut stream))?;
    println!();

    let t = &out.timing;
    eprintln!(
        "\n{} tokens on {} | final chain: {}",
        t.tokens_generated, out.transcript.env.backend, out.transcript.final_chain
    );
    if args.bench {
        eprintln!(
            "model load: {:.2?} | prompt eval: {:.2?} | decode: {:.2?} ({:.1} tok/s)\n\
             commitment work: {:.2?} = {:.3}% of inference",
            t.model_load,
            t.prompt_eval,
            t.decode,
            t.tokens_generated as f64 / t.decode.as_secs_f64(),
            t.commit,
            t.commit_overhead() * 100.0
        );
    }
    if let Some(path) = &args.commit {
        std::fs::write(path, serde_json::to_string_pretty(&out.transcript)?)?;
        eprintln!("wrote transcript to {}", path.display());
    }
    if let Some(trace) = &out.transcript.trace {
        eprintln!(
            "wrote trace ({} positions x {} cells) root {}",
            trace.n_positions,
            trace.n_layers + 1,
            trace.root
        );
    }
    Ok(())
}

fn cmd_replay(path: &Path) -> Result<()> {
    let transcript: Transcript = serde_json::from_str(&std::fs::read_to_string(path)?)
        .with_context(|| format!("parsing {path:?}"))?;
    match transcript.replay_chain() {
        ChainCheck::Ok => {
            eprintln!(
                "OK: chain replays cleanly over {} steps (final: {})",
                transcript.steps.len(),
                transcript.final_chain
            );
            Ok(())
        }
        ChainCheck::BadSeed { expected } => bail!("chain seed mismatch (recomputed {expected})"),
        ChainCheck::BadStep { index, expected } => {
            bail!("chain breaks at step {index} (recomputed {expected})")
        }
        ChainCheck::BadFinal { expected } => bail!("final chain mismatch (recomputed {expected})"),
    }
}

fn default_backend_tag() -> String {
    ZkBackend::Winterfell.tag().into()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DecodeProof {
    version: String,
    /// Proof system: "winterfell" (default, for pre-v0.5 proofs) or "halo2".
    #[serde(default = "default_backend_tag")]
    backend: String,
    step: u32,
    token_id: u32,
    vocab_size: u32,
    /// Circuit-friendly logits commitment (must equal the step's zk_digest).
    digest: String,
    /// Base64-encoded proof bytes.
    proof: String,
}

/// The generation hook that commits a step's logits, per backend.
fn build_zk_commit(
    prove_decode: bool,
    backend: ZkBackend,
) -> Result<Option<Box<vllm_infer::engine::ZkCommitFn>>> {
    use vllm_infer::engine::{ZkCommitFn, ZkCommitment};
    if !prove_decode {
        return Ok(None);
    }
    let f: Box<ZkCommitFn> = match backend {
        ZkBackend::Winterfell => Box::new(|q: &[i32]| {
            let salt = vllm_zk::random_salt();
            let digest = vllm_zk::commit_logits(q, salt).map_err(|e| e.to_string())?;
            Ok(ZkCommitment { salt, digest })
        }),
        ZkBackend::Halo2 => {
            #[cfg(feature = "zk-halo2")]
            {
                Box::new(halo2_backend::zk_commit)
            }
            #[cfg(not(feature = "zk-halo2"))]
            {
                bail!("the halo2 backend requires building with --features zk-halo2");
            }
        }
    };
    Ok(Some(f))
}

fn cmd_prove_decode(
    commitment: &Path,
    trace: &Path,
    step: u32,
    out: &Path,
    backend: ZkBackend,
) -> Result<()> {
    let transcript: Transcript = read_json(commitment)?;
    if transcript.replay_chain() != ChainCheck::Ok {
        bail!("transcript chain does not replay");
    }
    if transcript.sampler.mode != SamplerMode::Greedy {
        bail!("argmax proofs require a greedy transcript (top-p proofs are roadmap)");
    }
    let record = transcript
        .steps
        .get(step as usize)
        .with_context(|| format!("transcript has no step {step}"))?;
    let zk_digest = record
        .zk_digest
        .context("step has no zk commitment; generate with --prove-decode")?;

    let mut reader = vllm_core::trace::TraceReader::open(trace)?;
    let salts = reader
        .meta()
        .zk_salts
        .clone()
        .context("trace file has no zk salts; generate with --prove-decode")?;
    let salt = *salts.get(step as usize).context("no salt for step")?;
    let logits = reader.logits_row(step)?;

    // Recompute the digest under the selected backend; it must match the
    // chain-bound commitment (also rejects a backend/transcript mismatch),
    // then prove.
    let t0 = Instant::now();
    let (proof, version) = match backend {
        ZkBackend::Winterfell => {
            if vllm_zk::commit_logits(&logits, salt)? != zk_digest.0 {
                bail!("trace logits do not match the step's zk commitment (wrong --backend?)");
            }
            (
                vllm_zk::prove_argmax(&logits, salt, record.token_id)?,
                "vllm/zk-proof/v1",
            )
        }
        ZkBackend::Halo2 => (
            halo2_prove(&logits, salt, record.token_id, &zk_digest.0)?,
            "vllm/zk-proof-halo2/v1",
        ),
    };
    let envelope = DecodeProof {
        version: version.into(),
        backend: backend.tag().into(),
        step,
        token_id: record.token_id,
        vocab_size: transcript.vocab_size,
        digest: zk_digest.to_hex(),
        proof: vllm_core::protocol::b64_encode(&proof),
    };
    std::fs::write(out, serde_json::to_string_pretty(&envelope)?)?;
    eprintln!(
        "proved argmax ({}) for step {step} (token {}) in {:.2?}; {} KB -> {}",
        backend.tag(),
        record.token_id,
        t0.elapsed(),
        proof.len() / 1024,
        out.display()
    );
    Ok(())
}

fn cmd_verify_decode(commitment: &Path, proof_path: &Path) -> Result<()> {
    let transcript: Transcript = read_json(commitment)?;
    let envelope: DecodeProof = read_json(proof_path)?;
    let backend = match envelope.backend.as_str() {
        "winterfell" => ZkBackend::Winterfell,
        "halo2" => ZkBackend::Halo2,
        other => bail!("unknown decode-proof backend {other:?}"),
    };
    if transcript.replay_chain() != ChainCheck::Ok {
        bail!("transcript chain does not replay");
    }
    if transcript.sampler.mode != SamplerMode::Greedy {
        bail!("transcript is not greedy; argmax proof does not apply");
    }
    let record = transcript
        .steps
        .get(envelope.step as usize)
        .with_context(|| format!("transcript has no step {}", envelope.step))?;
    let zk_digest = record.zk_digest.context("step has no zk commitment")?;
    if envelope.digest != zk_digest.to_hex() {
        bail!("proof digest does not match the chain-bound zk commitment");
    }
    if envelope.token_id != record.token_id || envelope.vocab_size != transcript.vocab_size {
        bail!("proof claims do not match the transcript");
    }
    let proof_bytes =
        vllm_core::protocol::b64_decode(&envelope.proof).context("bad proof encoding")?;
    let t0 = Instant::now();
    match backend {
        ZkBackend::Winterfell => vllm_zk::verify_argmax(
            &zk_digest.0,
            record.token_id,
            transcript.vocab_size,
            &proof_bytes,
        )?,
        ZkBackend::Halo2 => halo2_verify(
            &zk_digest.0,
            record.token_id,
            transcript.vocab_size,
            &proof_bytes,
        )?,
    }
    eprintln!(
        "OK ({}): step {} emitted token {} = argmax of the committed logits ({:.2?})",
        backend.tag(),
        envelope.step,
        record.token_id,
        t0.elapsed()
    );
    Ok(())
}

// ---- halo2 formal-ZK backend (optional; heavy proof-system deps) ----

#[cfg(feature = "zk-halo2")]
fn halo2_prove(logits: &[i32], salt: [u64; 4], token: u32, expected: &[u8; 32]) -> Result<Vec<u8>> {
    if &halo2_backend::commit(logits, salt)? != expected {
        bail!("trace logits do not match the step's zk commitment (wrong --backend?)");
    }
    halo2_backend::prove(logits, salt, token)
}

#[cfg(not(feature = "zk-halo2"))]
fn halo2_prove(_: &[i32], _: [u64; 4], _: u32, _: &[u8; 32]) -> Result<Vec<u8>> {
    bail!("this proof is halo2; rebuild with --features zk-halo2 to prove it");
}

#[cfg(feature = "zk-halo2")]
fn halo2_verify(digest: &[u8; 32], token: u32, vocab: u32, proof: &[u8]) -> Result<()> {
    halo2_backend::verify(digest, token, vocab, proof)
}

#[cfg(not(feature = "zk-halo2"))]
fn halo2_verify(_: &[u8; 32], _: u32, _: u32, _: &[u8]) -> Result<()> {
    bail!("this proof is halo2; rebuild with --features zk-halo2 to verify it");
}

/// The halo2 formal-ZK backend uses a Poseidon commitment and a 32-byte salt;
/// it reuses the transcript's backend-agnostic zk-digest and salt slots (the
/// salt is stored as `[u64; 4]`, i.e. the same 32 bytes).
#[cfg(feature = "zk-halo2")]
mod halo2_backend {
    use anyhow::Result;
    use vllm_infer::engine::ZkCommitment;

    fn u64_to_bytes(s: [u64; 4]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, w) in s.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn bytes_to_u64(b: [u8; 32]) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (i, w) in out.iter_mut().enumerate() {
            *w = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        }
        out
    }

    /// Generation hook: commit a step's logits under a fresh salt.
    pub fn zk_commit(q: &[i32]) -> std::result::Result<ZkCommitment, String> {
        let salt = vllm_zk_halo2::random_salt();
        let digest = vllm_zk_halo2::commit_logits(q, &salt).map_err(|e| e.to_string())?;
        Ok(ZkCommitment {
            salt: bytes_to_u64(salt),
            digest,
        })
    }

    /// Recompute a step's digest from trace logits + stored salt.
    pub fn commit(logits: &[i32], salt: [u64; 4]) -> Result<[u8; 32]> {
        Ok(vllm_zk_halo2::commit_logits(logits, &u64_to_bytes(salt))?)
    }

    pub fn prove(logits: &[i32], salt: [u64; 4], token: u32) -> Result<Vec<u8>> {
        Ok(vllm_zk_halo2::prove_argmax(
            logits,
            &u64_to_bytes(salt),
            token,
        )?)
    }

    pub fn verify(digest: &[u8; 32], token: u32, vocab: u32, proof: &[u8]) -> Result<()> {
        vllm_zk_halo2::verify_argmax(digest, token, vocab, proof)?;
        Ok(())
    }
}
