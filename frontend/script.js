function handleSubmit() {
  const input = document.getElementById('userInput').value;
  const display = document.getElementById('displaySection');

  if (input.trim() === "") {
    display.innerHTML = "<p>Please enter something.</p>";
    return;
  }

  display.innerHTML = `<p><strong>You entered:</strong> ${input}</p>`;
}
