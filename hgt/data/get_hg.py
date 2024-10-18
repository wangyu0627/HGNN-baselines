import os
import pickle as pkl
import dgl
import torch

def process_acm():
    data_path = "acm"

    # paper_author
    paper_author_src = []
    paper_author_dst = []
    with open(os.path.join(data_path, "pa.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, author = int(_line[0]), int(_line[1])
            paper_author_src.append(paper)
            paper_author_dst.append(author)

    # paper_subject
    paper_subject_src = []
    paper_subject_dst = []
    with open(os.path.join(data_path, "ps.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, subject = int(_line[0]), int(_line[1])
            paper_subject_src.append(paper)
            paper_subject_dst.append(subject)

    # build graph
    G = dgl.heterograph(
        {
            ("paper", "pa", "author"): (paper_author_src, paper_author_dst),
            ("author", "ap", "paper"): (paper_author_dst, paper_author_src),
            ("paper", "ps", "subject"): (paper_subject_src, paper_subject_dst),
            ("subject", "sp", "paper"): (paper_subject_dst, paper_subject_src),
        }
    )

    print("Graph processed.")
    # save the processed data
    with open(os.path.join(data_path, "acm_hg.pkl"), "wb") as file:
        pkl.dump(G, file)

def process_freebase():
    data_path = "freebase"

    # movie_actor
    movie_actor_src = []
    movie_actor_dst = []
    with open(os.path.join(data_path, "ma.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            movie, actor = int(_line[0]), int(_line[1])
            movie_actor_src.append(movie)
            movie_actor_dst.append(actor)

    # movie_direct
    movie_direct_src = []
    movie_direct_dst = []
    with open(os.path.join(data_path, "md.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            movie, direct = int(_line[0]), int(_line[1])
            movie_direct_src.append(movie)
            movie_direct_dst.append(direct)

    # movie_writer
    movie_writer_src = []
    movie_writer_dst = []
    with open(os.path.join(data_path, "mw.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            movie, writer = int(_line[0]), int(_line[1])
            movie_writer_src.append(movie)
            movie_writer_dst.append(writer)

        # build graph
        G = dgl.heterograph(
            {
                ("movie", "ma", "actor"): (movie_actor_src, movie_actor_dst),
                ("actor", "am", "movie"): (movie_actor_dst, movie_actor_src),
                ("movie", "md", "direct"): (movie_direct_src, movie_direct_dst),
                ("direct", "dm", "movie"): (movie_direct_dst, movie_direct_src),
                ("movie", "mw", "writer"): (movie_writer_src, movie_writer_dst),
                ("writer", "wm", "movie"): (movie_writer_dst, movie_writer_src),
            }
        )
    print("Graph processed.")
    # save the processed data
    with open(os.path.join(data_path, "freebase_hg.pkl"), "wb") as file:
        pkl.dump(G, file)

def process_dblp():
    data_path = "dblp"

    # paper_author
    paper_author_src = []
    paper_author_dst = []
    with open(os.path.join(data_path, "pa.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, author = int(_line[0]), int(_line[1])
            paper_author_src.append(paper)
            paper_author_dst.append(author)

    # paper_conference
    paper_conference_src = []
    paper_conference_dst = []
    with open(os.path.join(data_path, "pc.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, conference = int(_line[0]), int(_line[1])
            paper_conference_src.append(paper)
            paper_conference_dst.append(conference)

    # paper_term
    paper_term_src = []
    paper_term_dst = []
    with open(os.path.join(data_path, "pt.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, term = int(_line[0]), int(_line[1])
            paper_term_src.append(paper)
            paper_term_dst.append(term)

    # build graph
    G = dgl.heterograph(
        {
            ("paper", "pa", "author"): (paper_author_src, paper_author_dst),
            ("author", "ap", "paper"): (paper_author_dst, paper_author_src),
            ("paper", "pc", "conference"): (paper_conference_src, paper_conference_dst),
            ("conference", "cp", "paper"): (paper_conference_dst, paper_conference_src),
            ("paper", "pt", "term"): (paper_term_src, paper_term_dst),
            ("term", "tp", "paper"): (paper_term_dst, paper_term_src),
        }
    )

    print("Graph processed.")
    # save the processed data
    with open(os.path.join(data_path, "dblp_hg.pkl"), "wb") as file:
        pkl.dump(G, file)

def process_aminer():
    data_path = "aminer"

    # paper_author
    paper_author_src = []
    paper_author_dst = []
    with open(os.path.join(data_path, "pa.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, author = int(_line[0]), int(_line[1])
            paper_author_src.append(paper)
            paper_author_dst.append(author)

    # paper_reference
    paper_reference_src = []
    paper_reference_dst = []
    with open(os.path.join(data_path, "pr.txt")) as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            paper, reference = int(_line[0]), int(_line[1])
            paper_reference_src.append(paper)
            paper_reference_dst.append(reference)

    # build graph
    G = dgl.heterograph(
        {
            ("paper", "pa", "author"): (paper_author_src, paper_author_dst),
            ("author", "ap", "paper"): (paper_author_dst, paper_author_src),
            ("paper", "pr", "reference"): (paper_reference_src, paper_reference_dst),
            ("reference", "rp", "paper"): (paper_reference_dst, paper_reference_src),
        }
    )

    print("Graph processed.")
    # save the processed data
    with open(os.path.join(data_path, "aminer_hg.pkl"), "wb") as file:
        pkl.dump(G, file)

if __name__ == '__main__':
    # process_acm()
    # process_freebase()
    process_dblp()
    # process_aminer()