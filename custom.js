function create_termynal() {
    Array.from(document.getElementsByClassName("termynal-block")).forEach(
        (element) => {
            new Termynal(`#${element.getAttribute("id")}`);
        }
    );
}

create_termynal();
