const essayInput = document.getElementById('essayInput');
const wordCount = document.getElementById('wordCount');
const warning = document.getElementById('warning');
const toast = document.getElementById('toast');
const shimmerScore = document.getElementById('shimmerScore');
const shimmerGen = document.getElementById('shimmerGen');
const audio = document.getElementById('typeSound');

essayInput.addEventListener('input', () => {
  const words = essayInput.value.trim().split(/\s+/).filter(Boolean);
  wordCount.textContent = `Words: ${words.length}`;
  warning.textContent = words.length > 250 ? "Over 250 words!" : "";
});

document.getElementById('scoreBtn').onclick = () => {
  document.getElementById('scoreSection').style.display = 'block';
  document.getElementById('generateSection').style.display = 'none';
};
document.getElementById('generateBtn').onclick = () => {
  document.getElementById('scoreSection').style.display = 'none';
  document.getElementById('generateSection').style.display = 'block';
};
document.getElementById('toggleTheme').onclick = () => {
  const body = document.body;
  const toggleBtn = document.getElementById('toggleTheme');
  body.classList.toggle('dark');
  toggleBtn.textContent = body.classList.contains('dark') ? '☀️' : '🌙';
};

document.getElementById('downloadBtn').onclick = () => {
  const text = document.getElementById('scoreResult').innerText;
  const blob = new Blob([text], { type: 'application/pdf' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'essay.pdf';
  link.click();
};

function showToast(msg) {
  toast.textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => toast.style.display = 'none', 2000);
}

function evaluateEssay() {
  shimmerScore.style.display = 'block';
  audio.play();
  setTimeout(() => {
    shimmerScore.style.display = 'none';
    const text = essayInput.value;
    const score = "Score: 7.0 / 9.0";
    const feedback = "Task Achievement: ✅\nCohesion: ✅\nLexical: ❌\nGrammar: ✅\n\nAvoid repetition like 'very'.";
    document.getElementById('scoreResult').innerText = score + "\n" + feedback;
    showToast("Scored!");
  }, 1500);
}

function generateEssay() {
  shimmerGen.style.display = 'block';
  audio.play();
  setTimeout(() => {
    shimmerGen.style.display = 'none';
    document.getElementById('generatedOutput').innerText = "Sure! Here is a general essay based on your prompt.\n\nIntroduction...\nBody Paragraph...\nConclusion...";
    showToast("Essay Generated");
  }, 1500);
}
