import { useState, type FormEvent } from "react";
import { Loader2, Search, Send } from "lucide-react";

interface SearchFormProps {
  isSubmitting: boolean;
  onSubmit: (query: string) => Promise<boolean>;
}

function SearchForm({ isSubmitting, onSubmit }: SearchFormProps) {
  const [input, setInput] = useState("");

  const submit = async () => {
    const query = input.trim();
    if (!query || isSubmitting) return;

    if (await onSubmit(query)) {
      setInput("");
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submit();
  };

  return (
    <section className="search-form-section">
      <div className="panel-section-title">
        <span>
          <Search size={13} />
          Retrieval query
        </span>
        <strong>API required</strong>
      </div>

      <form className="composer-box" onSubmit={handleSubmit}>
        <textarea
          rows={3}
          value={input}
          placeholder="Describe the lunar terrain to retrieve..."
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submit();
            }
          }}
          disabled={isSubmitting}
        />
        <button
          type="submit"
          className="send-button"
          disabled={!input.trim() || isSubmitting}
          aria-label="Run retrieval query"
          title="Run retrieval query"
        >
          {isSubmitting ? (
            <Loader2 size={16} className="spin" />
          ) : (
            <Send size={16} />
          )}
        </button>
      </form>
    </section>
  );
}

export default SearchForm;
