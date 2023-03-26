drop schema IF EXISTS companyDB; 
CREATE SCHEMA companyDB; 
USE companyDB; 


CREATE SCHEMA IF NOT EXISTS `companyDB` DEFAULT CHARACTER SET latin1 ;
USE `companyDB` ;

-- -----------------------------------------------------
-- Table `empfromacc`.`employee`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `companyDB`.`employee` (
  `ADDRESS` VARCHAR(50) NOT NULL,
  `BDATE` DATETIME NOT NULL,
  `DNO` SMALLINT(6) NOT NULL,
  `FNAME` VARCHAR(50) NOT NULL,
  `GENDER` VARCHAR(1) NOT NULL,
  `LNAME` VARCHAR(50) NOT NULL,
  `MINIT` VARCHAR(50) NULL DEFAULT NULL,
  `SALARY` DECIMAL(20,4) NOT NULL,
  `SSN` VARCHAR(9) NOT NULL,
  `SUPERSSN` VARCHAR(9) NULL DEFAULT NULL,
  PRIMARY KEY (`SSN`),
  INDEX `forKeyEmpUnary` (`SUPERSSN` ASC),
  CONSTRAINT `forKeyEmpUnary`
    FOREIGN KEY (`SUPERSSN`)
    REFERENCES `companyDB`.`employee` (`SSN`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = latin1;


-- -----------------------------------------------------
-- Table `empfromacc`.`department`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `companyDB`.`department` (
  `DNAME` VARCHAR(50) NOT NULL,
  `DNUMBER` SMALLINT(6) NOT NULL,
  `MGRSSN` VARCHAR(9) NOT NULL,
  `MGRSTARTDATE` DATETIME NOT NULL,
  PRIMARY KEY (`DNUMBER`),
  INDEX `fk_department_employee1_idx` (`MGRSSN` ASC),
  CONSTRAINT `fk_department_employee1`
    FOREIGN KEY (`MGRSSN`)
    REFERENCES `companyDB`.`employee` (`SSN`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = latin1;


-- -----------------------------------------------------
-- Table `empfromacc`.`dependent`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `companyDB`.`dependent` (
  `BDATE` DATETIME NOT NULL,
  `DEPENDENT_NAME` VARCHAR(50) NOT NULL,
  `ESSN` VARCHAR(9) NOT NULL,
  `GENDER` VARCHAR(1) NOT NULL,
  `RELATIONSHIP` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`ESSN`, `DEPENDENT_NAME`),
  CONSTRAINT `fk_dependent_employee1`
    FOREIGN KEY (`ESSN`)
    REFERENCES `companyDB`.`employee` (`SSN`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = latin1;


-- -----------------------------------------------------
-- Table `empfromacc`.`dept_locations`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `companyDB`.`dept_locations` (
  `DLOCATION` VARCHAR(50) NOT NULL,
  `DNUMBER` SMALLINT(6) NOT NULL,
  PRIMARY KEY (`DNUMBER`, `DLOCATION`),
  CONSTRAINT `fk_dept_locations_department1`
    FOREIGN KEY (`DNUMBER`)
    REFERENCES `companyDB`.`department` (`DNUMBER`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = latin1;


-- -----------------------------------------------------
-- Table `empfromacc`.`project`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `companyDB`.`project` (
  `DNUM` SMALLINT(6) NOT NULL,
  `PLOCATION` VARCHAR(50) NOT NULL,
  `PNAME` VARCHAR(50) NOT NULL,
  `PNUMBER` SMALLINT(6) NOT NULL,
  PRIMARY KEY (`PNUMBER`),
  INDEX `fk_project_department1_idx` (`DNUM` ASC),
  CONSTRAINT `fk_project_department1`
    FOREIGN KEY (`DNUM`)
    REFERENCES `companyDB`.`department` (`DNUMBER`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = latin1;


-- -----------------------------------------------------
-- Table `empfromacc`.`works_on`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `companyDB`.`works_on` (
  `ESSN` VARCHAR(9) NOT NULL,
  `HOURS` DOUBLE NULL DEFAULT NULL,
  `PNO` SMALLINT(6) NOT NULL,
  PRIMARY KEY (`ESSN`, `PNO`),
  INDEX `fk_works_on_project1_idx` (`PNO` ASC),
  CONSTRAINT `fk_works_on_project1`
    FOREIGN KEY (`PNO`)
    REFERENCES `companyDB`.`project` (`PNUMBER`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_works_on_employee1`
    FOREIGN KEY (`ESSN`)
    REFERENCES `companyDB`.`employee` (`SSN`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = latin1;

# upto this point we have created the tables and relationships. 





  

SET foreign_key_checks = 0; 

-- inserting data
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('John', 'B', 'Smith', '123456789', STR_TO_DATE ('01-09-1965','%m-%d-%Y'), '731 Fondren, Houston, TX', 'M', 30000, '333445555', 5);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('Franklin', 'T', 'Wong', '333445555', STR_TO_DATE ('12-08-1955','%m-%d-%Y'), '638 voss,Houston, TX', 'M', 40000, '888665555', 5);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('Joyce', 'A', 'English', '453453453', STR_TO_DATE ('07-31-1972','%m-%d-%Y'), '5631 Rice, Houston, TX', 'F', 25000, '333445555', 5);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('Ramesh', 'K', 'Narayan', '666884444', STR_TO_DATE ('09-15-1962','%m-%d-%Y'), '975 Fire Oak, Humble, TX', 'M', 38000, '333445555', 5);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('James', 'E', 'Borg', '888665555', STR_TO_DATE ('11-10-1937','%m-%d-%Y'), '450 Stone, Houston, TX', 'M', 55000, NULL, 1);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('Jennifer', 'S', 'Wallace', '987654321', STR_TO_DATE ('06-20-1941','%m-%d-%Y'), '291Berry, Bellair, TX', 'F', 43000, '888665555', 4);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('Ahmad', 'V', 'Jabbar', '987987987', STR_TO_DATE ('03-29-1969','%m-%d-%Y'), '980 Dallas, Houston, TX', 'M', 25000, '987654321', 4);
INSERT INTO EMPLOYEE (FNAME, MINIT, LNAME, SSN, BDATE, ADDRESS, GENDER, SALARY, SUPERSSN, DNO)
VALUES ('Alicia', 'J', 'Zelaya', '999887777', STR_TO_DATE ('07-19-1968','%m-%d-%Y'), '3321 Castle, Spring, TX', 'F', 25000, '987654321', 4);


  
# project table
-- inserting data
INSERT INTO PROJECT (PNAME, PNUMBER, PLOCATION, DNUM)
VALUES ('ProductX', 1, 'Bellair', 5);
INSERT INTO PROJECT (PNAME, PNUMBER, PLOCATION, DNUM)
VALUES ('ProductY', 2, 'Sugarland', 5);
INSERT INTO PROJECT (PNAME, PNUMBER, PLOCATION, DNUM)
VALUES ('ProductZ', 3, 'Houston', 5);
INSERT INTO PROJECT (PNAME, PNUMBER, PLOCATION, DNUM)
VALUES ('Computerization', 10, 'Stafford', 4);
INSERT INTO PROJECT (PNAME, PNUMBER, PLOCATION, DNUM)
VALUES ('Reorganization', 20, 'Houston', 1);
INSERT INTO PROJECT (PNAME, PNUMBER, PLOCATION, DNUM)
VALUES ('Newbenefits', 30, 'Stafford', 4);


# works_on
-- inserting data
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('123456789', 1, 32.5);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('123456789', 2, 7.5);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('333445555', 2, 10);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('333445555', 3, 10);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('333445555', 10, 10);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('333445555', 20, 10);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('453453453', 1, 20);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('453453453', 2, 20);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('666884444', 3, 40);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('888665555', 20, NULL);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('987654321', 20, 15);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('987654321', 30, 20);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('987987987', 10, 35);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('987987987', 30, 5);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('999887777', 10, 10);
INSERT INTO WORKS_ON (ESSN, PNO, HOURS)
VALUES ('999887777', 30, 30);


# department 
-- inserting data
INSERT INTO DEPARTMENT (DNAME, DNUMBER, MGRSSN, MGRSTARTDATE)
VALUES ('Headquarters', 1, '888665555', STR_TO_DATE ('06-19-1981','%m-%d-%Y'));
INSERT INTO DEPARTMENT (DNAME, DNUMBER, MGRSSN, MGRSTARTDATE)
VALUES ('Administration', 4, '987654321', STR_TO_DATE ('01-01-1995','%m-%d-%Y'));
INSERT INTO DEPARTMENT (DNAME, DNUMBER, MGRSSN, MGRSTARTDATE)
VALUES ('Research', 5, '333445555', STR_TO_DATE ('05-22-1988','%m-%d-%Y'));


# dependent
-- inserting data
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('123456789', 'Alice', 'F', STR_TO_DATE ('12-30-1988','%m-%d-%Y'), 'Daughter');
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('123456789', 'Elizabeth', 'F', STR_TO_DATE ('05-05-1967','%m-%d-%Y'), 'Spouse');
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('123456789', 'Michael', 'M', STR_TO_DATE ('01-04-1988','%m-%d-%Y'), 'Son');
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('333445555', 'Alice', 'F', STR_TO_DATE ('04-05-1986','%m-%d-%Y'), 'Daughter');
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('333445555', 'Joy', 'F', STR_TO_DATE ('05-03-1958','%m-%d-%Y'), 'Spouse');
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('333445555', 'Theodore', 'M', STR_TO_DATE ('10-20-1983','%m-%d-%Y'), 'Son');
INSERT INTO DEPENDENT (ESSN, DEPENDENT_NAME, GENDER, BDATE, RELATIONSHIP)
VALUES ('987654321', 'Abner', 'M', STR_TO_DATE ('02-28-1942','%m-%d-%Y'), 'Spouse');


# dept locations
-- inserting data
INSERT INTO DEPT_LOCATIONS (DNUMBER, DLOCATION)
VALUES (1, 'Houston');
INSERT INTO DEPT_LOCATIONS (DNUMBER, DLOCATION)
VALUES (4, 'Stafford');
INSERT INTO DEPT_LOCATIONS (DNUMBER, DLOCATION)
VALUES (5, 'Bellair');
INSERT INTO DEPT_LOCATIONS (DNUMBER, DLOCATION)
VALUES (5, 'Houston');
INSERT INTO DEPT_LOCATIONS (DNUMBER, DLOCATION)
VALUES (5, 'Sugarland');

SET foreign_key_checks = 1; 