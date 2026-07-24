import { useState, type FormEvent } from "react";
import { Loader2, Search, Send } from "lucide-react";

interface SearchFormProps {
  isSubmitting: boolean;
  onSubmit: (query: string) => Promise<boolean>;
  placement?: "initial" | "conversation";
}

function SearchForm({
  isSubmitting,
  onSubmit,
  placement = "initial",
}: SearchFormProps) {
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
    <section className={`search-form-section ${placement}`}>
      <div className="panel-section-title">
        <span>
          <Search size={13} />
          {placement === "initial" ? "Start a search" : "Next query"}
        </span>
        <strong>Semantic retrieval</strong>
      </div>

      <form className="composer-box" onSubmit={handleSubmit}>
        <textarea
          rows={placement === "initial" ? 4 : 2}
          value={input}
          maxLength={500}
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
