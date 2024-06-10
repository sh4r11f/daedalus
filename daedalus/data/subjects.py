# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: subjects.py
#
#
#          DESCRIPTION: Database of subjects and their parameters
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 06-10-2024-[78 105 98 105114117]
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, CheckConstraint, UniqueConstraint, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Base class for our ORM models
Base = declarative_base()


class Subject(Base):
    """
    This class represents a Subject in the database.

    Attributes:
        id (int): The primary key.
        initials (str): The initials of the subject.
        name (str): The name of the subject.
        age (int): The age of the subject.
        gender (str): The gender of the subject.
        netid (str): The network ID of the subject.
        email (str): The email of the subject.
        vision (str): The vision status of the subject.
        dominant_eye (str): The dominant eye of the subject.
        dominant_hand (str): The dominant hand of the subject.
    """
    __tablename__ = 'subjects'

    id = Column(Integer, primary_key=True)
    initials = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    vision = Column(String, nullable=False)
    name = Column(String)
    netid = Column(String)
    email = Column(String)
    dominant_eye = Column(String)
    dominant_hand = Column(String)
    date = Column(DateTime, default=datetime.now)

    # Parents
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    experiments = relationship('Experiment', back_populates='subjects')

    # Children
    tasks = relationship('Task', secondary='subject_tasks', back_populates='subjects')

    __table_args__ = (
        CheckConstraint("age > 0", name="age_check"),
        CheckConstraint("gender in ('Male', 'Female', 'NB')", name="gender_check"),
        CheckConstraint("vision in ('Normal', 'Corrected', 'Impaired')", name="vision_check"),
        CheckConstraint("dominant_eye in ('Right', 'Left')", name="dominant_eye_check"),
        CheckConstraint("dominant_hand in ('Right', 'Left', 'Both')", name="dominant_hand_check"),
        UniqueConstraint('initials', 'age', 'gender', name='unique_subject')
    )

    def __repr__(self):
        return f'<Subject(id={self.id}, initials={self.initials}, name={self.name}, age={self.age})>'

