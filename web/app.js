/**
 * AppleSupport Agent — Light Premium AI SaaS Frontend Controller
 * Smooth interactions, dynamic backend rendering, and copy functionality.
 */

(function () {
  'use strict';

  // DOM Elements
  const tweetInput = document.getElementById('tweetInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const errorAlert = document.getElementById('errorAlert');
  const errorMessage = document.getElementById('errorMessage');
  const examplesList = document.getElementById('examplesList');

  const resultsStack = document.getElementById('resultsStack');

  // Decision Card
  const decisionBadge = document.getElementById('decisionBadge');
  const decisionIcon = document.getElementById('decisionIcon');
  const decisionLabel = document.getElementById('decisionLabel');
  const decisionReason = document.getElementById('decisionReason');

  // Intent Card
  const detectedIntent = document.getElementById('detectedIntent');
  const confidencePercentage = document.getElementById('confidencePercentage');
  const confidenceBar = document.getElementById('confidenceBar');

  // Retrieved Context Card
  const similarityScore = document.getElementById('similarityScore');
  const similarCustomerTweet = document.getElementById('similarCustomerTweet');
  const historicalSupportResponse = document.getElementById('historicalSupportResponse');
  const caseIndexIndicator = document.getElementById('caseIndexIndicator');
  const prevCaseBtn = document.getElementById('prevCaseBtn');
  const nextCaseBtn = document.getElementById('nextCaseBtn');

  // Generated Reply Card
  const generatedReplyContainer = document.getElementById('generatedReplyContainer');
  const copyReplyBtn = document.getElementById('copyReplyBtn');
  const copyBtnLabel = document.getElementById('copyBtnLabel');

  // State
  let currentRetrievedCases = [];
  let currentCaseIndex = 0;
  let currentGeneratedReply = '';
  let isAnalyzing = false;

  function init() {
    // Analyze button click
    analyzeBtn.addEventListener('click', triggerAnalysis);

    // Ctrl/Cmd + Enter trigger
    tweetInput.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        triggerAnalysis();
      }
    });

    // Example chips click
    if (examplesList) {
      examplesList.addEventListener('click', (e) => {
        const chip = e.target.closest('.example-chip');
        if (!chip) return;
        const query = chip.getAttribute('data-query');
        if (query) {
          tweetInput.value = query;
          hideError();
          tweetInput.focus();
        }
      });
    }

    // Copy Reply button click
    copyReplyBtn.addEventListener('click', handleCopyReply);

    // Case pagination controls
    prevCaseBtn.addEventListener('click', () => {
      if (currentCaseIndex > 0) {
        currentCaseIndex--;
        renderCurrentCase();
      }
    });

    nextCaseBtn.addEventListener('click', () => {
      if (currentCaseIndex < currentRetrievedCases.length - 1) {
        currentCaseIndex++;
        renderCurrentCase();
      }
    });

    tweetInput.addEventListener('input', hideError);
  }

  function showError(msg) {
    errorMessage.textContent = msg || 'Unable to analyze message. Please try again.';
    errorAlert.style.display = 'flex';
  }

  function hideError() {
    errorAlert.style.display = 'none';
  }

  function setLoading(loading) {
    isAnalyzing = loading;
    analyzeBtn.disabled = loading;
    const btnContent = analyzeBtn.querySelector('.btn-content');
    const btnSpinner = analyzeBtn.querySelector('.btn-spinner');

    if (loading) {
      btnContent.style.display = 'none';
      btnSpinner.style.display = 'inline-flex';
    } else {
      btnContent.style.display = 'inline-flex';
      btnSpinner.style.display = 'none';
    }
  }

  async function triggerAnalysis() {
    const rawMessage = tweetInput.value.trim();
    if (!rawMessage) {
      showError('Please enter a customer message or select an example above.');
      tweetInput.focus();
      return;
    }

    hideError();
    setLoading(true);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ message: rawMessage })
      });

      if (!response.ok) {
        let errDetail = `Server returned HTTP ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.error) errDetail = errData.error;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      showError(`Analysis failed: ${err.message || 'Server error or backend unavailable.'}`);
    } finally {
      setLoading(false);
    }
  }

  function renderResults(data) {
    if (!data) return;

    // 1. Decision Card
    const decision = (data.decision || 'UNKNOWN').toUpperCase();
    const reason = data.escalation_reason || 'Historical support cases provide sufficient evidence for this request.';

    decisionBadge.className = 'status-badge';
    if (decision === 'AUTO_HANDLE' || decision === 'AUTO-HANDLE') {
      decisionBadge.classList.add('auto-handle');
      decisionIcon.textContent = '✓';
      decisionLabel.textContent = 'AUTO-HANDLE';
    } else {
      decisionBadge.classList.add('escalate');
      decisionIcon.textContent = '⚠';
      decisionLabel.textContent = 'ESCALATE';
    }
    decisionReason.textContent = reason;

    // 2. Intent Classification Card
    detectedIntent.textContent = data.predicted_intent || 'other_general_inquiry';

    let confRaw = data.confidence;
    if (typeof confRaw !== 'number') confRaw = parseFloat(confRaw) || 0.0;
    const confPct = Math.max(0, Math.min(100, confRaw > 1 ? confRaw : confRaw * 100));
    const confStr = `${confPct.toFixed(1)}%`;

    confidencePercentage.textContent = confStr;
    // Reset bar width then animate to target
    confidenceBar.style.width = '0%';
    setTimeout(() => {
      confidenceBar.style.width = confStr;
    }, 50);

    // 3. Retrieved Context Card
    currentRetrievedCases = data.retrieved_cases || [];
    currentCaseIndex = 0;
    renderCurrentCase();

    // 4. Generated Reply Card
    currentGeneratedReply = data.drafted_reply || 'Thank you for reaching out to Apple Support.';
    generatedReplyContainer.textContent = currentGeneratedReply;
    resetCopyBtn();

    // Reveal Results Stack smoothly
    resultsStack.style.display = 'flex';

    // Smooth scroll into view on small screens
    if (window.innerWidth < 768) {
      resultsStack.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function renderCurrentCase() {
    if (!currentRetrievedCases || currentRetrievedCases.length === 0) {
      similarityScore.textContent = 'N/A';
      caseIndexIndicator.textContent = 'No cases found';
      similarCustomerTweet.textContent = '"No similar historical inquiry found."';
      historicalSupportResponse.textContent = '"No historical resolution found."';
      prevCaseBtn.disabled = true;
      nextCaseBtn.disabled = true;
      return;
    }

    const item = currentRetrievedCases[currentCaseIndex];
    let simScore = item.similarity;
    if (typeof simScore !== 'number') simScore = parseFloat(simScore) || 0.0;
    const simPct = (simScore > 1 ? simScore : simScore * 100).toFixed(1);

    similarityScore.textContent = `${simPct}% Match`;
    caseIndexIndicator.textContent = `Case ${currentCaseIndex + 1} of ${currentRetrievedCases.length}`;

    similarCustomerTweet.textContent = `"${item.customer_message || ''}"`;
    historicalSupportResponse.textContent = `"${item.support_reply || ''}"`;

    prevCaseBtn.disabled = currentCaseIndex === 0;
    nextCaseBtn.disabled = currentCaseIndex === currentRetrievedCases.length - 1;
  }

  async function handleCopyReply() {
    if (!currentGeneratedReply) return;

    try {
      await navigator.clipboard.writeText(currentGeneratedReply);
      copyReplyBtn.classList.add('copied');
      copyBtnLabel.textContent = 'Copied ✓';
      setTimeout(() => resetCopyBtn(), 2000);
    } catch (_) {
      const tempArea = document.createElement('textarea');
      tempArea.value = currentGeneratedReply;
      document.body.appendChild(tempArea);
      tempArea.select();
      try {
        document.execCommand('copy');
        copyReplyBtn.classList.add('copied');
        copyBtnLabel.textContent = 'Copied ✓';
        setTimeout(() => resetCopyBtn(), 2000);
      } catch (err) {
        console.error('Failed to copy reply text:', err);
      } finally {
        document.body.removeChild(tempArea);
      }
    }
  }

  function resetCopyBtn() {
    copyReplyBtn.classList.remove('copied');
    copyBtnLabel.textContent = 'Copy Reply';
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
